//
// Created by keega on 9/18/2021.
//

#include "sim/search/BattleSearcher.h"
#include "sim/search/ExpertKnowledge.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <string>
#include <memory>
#include <unordered_set>

using namespace sts;

std::int64_t simulationIdx = 0; // for debugging

// Telemetry category for a chance node's stochastic action: 0 = END_TURN (monster rolls +
// start-of-turn draws), 1 = card play, 2 = other (potions, card selects).
static int chanceCat(const search::Action &a) {
    switch (a.getActionType()) {
        case search::ActionType::END_TURN: return 0;
        case search::ActionType::CARD:     return 1;
        default:                           return 2;
    }
}


namespace sts::search {
    thread_local search::BattleSearcher *g_debug_scum_search;

    // Arbitrary nonzero seed for the state hash accumulator.
    constexpr std::uint64_t HASH_SEED = 14695981039346656037ULL;

    // Mix one whole word into the accumulator (MurmurHash3 x64 body, one 64-bit lane). This
    // touches the value in a handful of ops, with strong avalanche. The hash is used only to bucket
    // nodes in stateToNode -- collisions are resolved exactly by equalForSearch -- so a different
    // hash value changes only the bucketing performance, never the node graph or the search result.
    inline void hash_combine(std::uint64_t& hash, std::uint64_t value) {
        constexpr std::uint64_t c1 = 0x87c37b91114253d5ULL;
        constexpr std::uint64_t c2 = 0x4cf5ad432745937fULL;
        value *= c1;
        value = (value << 31) | (value >> 33);
        value *= c2;
        hash ^= value;
        hash = (hash << 27) | (hash >> 37);
        hash = hash * 5 + 0x52dce729ULL;
    }

    inline void hash_combine(std::uint64_t& hash, std::uint32_t value) {
        hash_combine(hash, static_cast<std::uint64_t>(value));
    }

    inline void hash_combine(std::uint64_t& hash, int value) {
        hash_combine(hash, static_cast<std::uint64_t>(static_cast<std::uint32_t>(value)));
    }

    inline void hash_combine(std::uint64_t& hash, bool value) {
        hash_combine(hash, static_cast<std::uint64_t>(value ? 1u : 0u));
    }

    // Hash BattleContext state for graph search deduplication
    // Only hashes observable game state, not internal RNG or debug counters
    std::size_t hashBattleState(const BattleContext& bc) {
        std::uint64_t hash = HASH_SEED;

        // Hash turn and outcome
        hash_combine(hash, bc.turn);
        hash_combine(hash, static_cast<int>(bc.outcome));
        hash_combine(hash, static_cast<int>(bc.inputState));

        // Hash player state
        const auto& p = bc.player;
        hash_combine(hash, p.curHp);
        hash_combine(hash, p.maxHp);
        hash_combine(hash, p.block);
        hash_combine(hash, p.energy);
        hash_combine(hash, p.cardsPlayedThisTurn);
        hash_combine(hash, p.statusBits0);
        hash_combine(hash, p.statusBits1);
        hash_combine(hash, p.strength);
        hash_combine(hash, p.dexterity);

        // Hash monsters
        for (int i = 0; i < bc.monsters.monsterCount; ++i) {
            const auto& m = bc.monsters.arr[i];
            hash_combine(hash, static_cast<int>(m.id));
            hash_combine(hash, m.curHp);
            hash_combine(hash, m.maxHp);
            hash_combine(hash, m.block);
            hash_combine(hash, m.statusBits);
            hash_combine(hash, static_cast<int>(m.moveHistory[0]));
            // Pending vs already-rolled intent are different information sets (Runic Dome).
            // The rollInputs snapshot is left to equalForSearch's exact comparison.
            hash_combine(hash, static_cast<int>(m.pendingMoveRolls));
        }

        // Hash the hand as a multiset: combine per-card hashes COMMUTATIVELY (sum) so
        // permutations hash identically without sorting a scratch copy (this is the hot path:
        // every getOrCreateNode hashes). equalForSearch compares the hand as a multiset to
        // match; extra cross-multiset collisions from the commutative fold are resolved there.
        hash_combine(hash, bc.cards.cardsInHand);
        {
            std::uint64_t handAcc = 0;
            for (int i = 0; i < bc.cards.cardsInHand; ++i) {
                const auto& c = bc.cards.hand[i];
                std::uint64_t ch = 0;
                hash_combine(ch, static_cast<int>(c.id));
                hash_combine(ch, c.cost);
                hash_combine(ch, c.costForTurn);
                hash_combine(ch, c.upgraded);
                handAcc += ch;
            }
            hash_combine(hash, handAcc);
        }

        // Hash draw pile with position (the sorted unknown region hashes canonically; the known
        // top/bottom stacks are position-significant). The known counts distinguish a known top
        // [A,B] from an unknown {A,B} — different information sets even with equal contents.
        hash_combine(hash, bc.cards.drawPile.knownCount());
        hash_combine(hash, bc.cards.drawPile.knownBottom());
        hash_combine(hash, static_cast<int>(bc.cards.drawPile.size()));
        for (int i = 0; i < static_cast<int>(bc.cards.drawPile.size()); ++i) {
            hash_combine(hash, i);
            hash_combine(hash, static_cast<int>(bc.cards.drawPile[i].id));
        }

        // Hash discard pile (including position for ordering)
        hash_combine(hash, static_cast<int>(bc.cards.discardPile.size()));
        int discardIdx = 0;
        for (const auto& c : bc.cards.discardPile) {
            hash_combine(hash, discardIdx++); // Position in discard
            hash_combine(hash, static_cast<int>(c.id));
        }

        // Hash exhaust pile contents (iteration order is order-sensitive via the accumulator)
        hash_combine(hash, static_cast<int>(bc.cards.exhaustPile.size()));
        for (const auto& c : bc.cards.exhaustPile) {
            hash_combine(hash, static_cast<int>(c.id));
        }

        // Hash potions
        hash_combine(hash, bc.potionCount);
        for (int i = 0; i < bc.potionCount; ++i) {
            hash_combine(hash, static_cast<int>(bc.potions[i]));
        }

        return hash;
    }
}



search::BattleSearcher::BattleSearcher(const BattleContext &bc, search::EvalFnc _evalFnc)
    : rootState(new BattleContext(bc)), evalFnc(std::move(_evalFnc)), randGen(bc.seed+bc.floorNum), rolloutAgent(true, bc.seed+bc.floorNum) {
    // Open-addressed dedup table, sized once. 64k slots * 16B = 1 MB; comfortably holds any
    // sensible simulationCountBase while keeping the load factor low so probe chains stay short.
    constexpr std::size_t dedupSlots = std::size_t(1) << 16;
    stateToNode.assign(dedupSlots, DedupSlot{0, nullptr});
    stateToNodeMask = dedupSlots - 1;
    // Allocate the root + populate dedup state for one-off `BattleSearcher(bc); search(N);` callers
    // (test.cpp, pybind). playoutBattle's reuse-across-decisions path overrides via rerootAt later.
    resetForSearch();
}

search::BattleSearcher::~BattleSearcher() = default;

void search::BattleSearcher::setRoot(const BattleContext &bc) {
    *rootState = bc;
    // Reseed exactly as the constructor does, so each decision's rollouts are independent of the
    // previous decision's -- matching the old behavior of constructing a fresh searcher per move.
    randGen.seed(bc.seed + bc.floorNum);
    rolloutAgent = SimpleAgent(true, bc.seed + bc.floorNum);
    resetForSearch();
}

void search::BattleSearcher::rerootAt(Node* newRoot) {
    // Tree reuse: leave the node pool + dedup table alone (the chosen subtree's cached visits and
    // evals serve as a head start for the new search). Just redirect root and reseed rng so the
    // new decision's rollouts behave the same as if we'd constructed a fresh searcher at this state.
    root = newRoot;
    *rootState = newRoot->state;
    randGen.seed(rootState->seed + rootState->floorNum);
    rolloutAgent = SimpleAgent(true, rootState->seed + rootState->floorNum);
}

// Claims a node from the pool: recycles allNodes[poolUsed] (resetting its bookkeeping but keeping
// its allocated state/edges storage) or grows the pool. Node addresses are stable because allNodes
// holds unique_ptrs, so edges may keep raw pointers across reallocations of the outer vector.
search::BattleSearcher::Node* search::BattleSearcher::allocNode() {
    if (poolUsed < allNodes.size()) {
        Node* n = allNodes[poolUsed++].get();
        n->edges.clear();
        n->simulationCount = 0;
        n->evaluationSum = 0;
        n->isRandomNode = false;
        n->parent = nullptr;
        n->outcomesGenerated = 0;
        n->randomnessBase = 0;
        n->stochasticAction = Action{};
        return n;
    }
    allNodes.push_back(std::make_unique<Node>());
    ++poolUsed;
    return allNodes.back().get();
}

// Moves `state` into the node only when a new node is created; on a match `state` is left
// intact, so callers may still read it.
search::BattleSearcher::Node* search::BattleSearcher::getOrCreateNode(BattleContext &&state) {
    const std::size_t hash = search::hashBattleState(state);

    // Linear probe the open-addressed table; equalForSearch resolves hash collisions exactly.
    std::size_t i = hash & stateToNodeMask;
    while (stateToNode[i].node != nullptr) {
        if (stateToNode[i].hash == hash && stateToNode[i].node->state.equalForSearch(state)) {
            lastNodeWasCreated = false;
            return stateToNode[i].node;
        }
        i = (i + 1) & stateToNodeMask;
    }

    Node* newNode = allocNode();
    // Swap rather than move-assign: the caller's scratch buffer inherits this recycled node's
    // old pile/edge vector capacities instead of being emptied, so the steady-state expansion
    // loop performs no heap allocation at all.
    std::swap(newNode->state, state);
    stateToNode[i] = {hash, newNode};
    lastNodeWasCreated = true;
    ++stats.nodesCreated;
    return newNode;
}

bool search::BattleSearcher::resetForSearch() {
    g_debug_scum_search = this;

    // Recycle the pool rather than freeing it: reclaim every node for reuse and drop the dedup map.
    poolUsed = 0;
    std::memset(stateToNode.data(), 0, stateToNode.size() * sizeof(DedupSlot));
    root = allocNode();
    root->state = *rootState;

    if (isTerminalState(root->state)) {
        root->evaluationSum = evaluateEndState(root->state);
        root->simulationCount = 1;
        return false;
    }
    return true;
}

void search::BattleSearcher::search(int64_t simulations) {
    g_debug_scum_search = this;
    // Root setup (setRoot/rerootAt) has run; if root is already terminal we have nothing to do.
    if (isTerminalState(root->state)) return;

    for (std::int64_t simCount = 0; simCount < simulations; ++simCount) {
        step();
    }
}

void search::BattleSearcher::searchForMicros(int64_t maxMicros) {
    g_debug_scum_search = this;
    if (isTerminalState(root->state)) return;

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(maxMicros);

    // Amortize the clock read over a small batch of steps. A single step (one tree descent plus a
    // rollout to a terminal state) ranges from well under a microsecond to tens of microseconds, so
    // a batch this size keeps both the now() overhead and the overshoot past the deadline negligible
    // relative to any usable time budget. The leading check makes a zero/expired budget a no-op.
    constexpr int stepsPerTimeCheck = 16;
    while (std::chrono::steady_clock::now() < deadline) {
        for (int i = 0; i < stepsPerTimeCheck; ++i) {
            step();
        }
    }
}

void search::BattleSearcher::step() {
    ++stats.steps;
    searchStack.clear();
    searchStack.push_back(root);
    actionStack.clear();

    // Cycle guard: is `n` already on the descent path? Linear scan; depth is small in practice.
    auto onPath = [&](Node* n) {
        return std::find(searchStack.begin(), searchStack.end(), n) != searchStack.end();
    };

    Node* cur = root;
    Edge* stagedOutcomeEdge = nullptr;  // outcome 0 of a chance node created this iteration, already executed
    int depth = 0;
    while (true) {
        // turn / cardsPlayedThisTurn are NOT globally monotonic -- potion use and card-select
        // actions advance neither -- so the transposition graph can genuinely cycle. The known
        // case is drinking Entropic Brew, which can re-roll itself: a stochastic action whose
        // outcome is gameplay-identical to the pre-action state (differing only in rng, which
        // equalForSearch ignores), deduping the outcome back onto the path. The onPath guards
        // below break such cycles by rolling out instead of descending into an on-path node; this
        // bound is only a backstop for any cycle they fail to catch.
        if (++depth > 5000) {
            throw std::runtime_error("BattleSearcher::step descent exceeded depth bound");
        }

        if (cur->isRandomNode) {
            // Chance node: resolve one outcome (staged at creation, or sampled/selected), descend into it.
            Edge* outcomeEdge;
            if (stagedOutcomeEdge != nullptr) {
                outcomeEdge = stagedOutcomeEdge;
                stagedOutcomeEdge = nullptr;
            } else {
                outcomeEdge = selectChanceOutcome(*cur);
            }
            ++outcomeEdge->visitCount;
            Node* child = outcomeEdge->node;

            if (onPath(child)) {
                // Outcome reproduces a state already on the path -> cycle. Roll out instead.
                BattleContext &rollout = (rolloutScratch = child->state);
                rolloutToEnd(rollout, actionStack);
                updateFromPlayout(searchStack, actionStack, rollout);
                return;
            }

            searchStack.push_back(child);

            if (child->simulationCount == 0) {
                BattleContext &rollout = (rolloutScratch = child->state);
                rolloutToEnd(rollout, actionStack);
                updateFromPlayout(searchStack, actionStack, rollout);
                return;
            }
            cur = child;
            continue;
        }

        if (isTerminalState(cur->state)) {
            updateFromPlayout(searchStack, actionStack, cur->state);
            return;
        }

        if (cur->edges.empty()) {
            ++simulationIdx;
            enumerateActionsForNode(*cur, cur->state);
            if (cur->edges.empty()) {
                // No legal actions but not terminal: nothing to do but evaluate.
                updateFromPlayout(searchStack, actionStack, cur->state);
                return;
            }
        }

        const int selectIdx = selectBestEdgeToSearch(*cur);
        Edge &edge = cur->edges[selectIdx];
        ++edge.visitCount;

        if (edge.node != nullptr) {
            // Already expanded: descend (transposition or revisit).
            if (onPath(edge.node)) {
                // Descending would revisit an on-path node -> cycle. Roll out instead.
                BattleContext &rollout = (rolloutScratch = edge.node->state);
                rolloutToEnd(rollout, actionStack);
                updateFromPlayout(searchStack, actionStack, rollout);
                return;
            }
            actionStack.push_back(edge.action);
            cur = edge.node;
            searchStack.push_back(cur);
            continue;
        }

        // First traversal of this action: execute on a copy of the current state. The rng is
        // pre-seeded to Random(base + 0) so that, if the action turns out to be stochastic,
        // this execution is exactly chance outcome 0 and can be kept rather than redone.
        BattleContext &next = (expandScratch = cur->state);
        const Random preActionRng = next.rng;
        Random baseGen = preActionRng;
        const auto randomnessBase = static_cast<std::uint64_t>(baseGen.randomLong());
        next.rng = Random(randomnessBase);
        edge.action.execute(next);
        actionStack.push_back(edge.action);
        const bool rngChanged = next.rng.counter != 0;

        if (rngChanged) {
            // Stochastic action: create a chance node that sources its pre-action state
            // from `cur` and resolves outcomes via Random(randomnessBase + N). `next` already
            // holds outcome 0; record it as the chance node's first edge and stage that edge
            // for the descent at the top of the next loop iteration.
            Node* chance = allocNode();
            chance->isRandomNode = true;
            chance->stochasticAction = edge.action;
            chance->parent = cur;
            chance->randomnessBase = randomnessBase;
            chance->outcomesGenerated = 1;
            edge.node = chance;

            ++stats.chanceOutcomesSampled;
            ++stats.wSampled[chanceCat(edge.action)];
            Edge outcomeEdge;
            outcomeEdge.action = Action{};
            outcomeEdge.node = getOrCreateNode(std::move(next));
            if (!lastNodeWasCreated) ++stats.chanceTranspositions;  // fresh chance node: no siblings yet
            outcomeEdge.rngAdvanceSteps = 0;
            chance->edges.push_back(outcomeEdge);
            stagedOutcomeEdge = &chance->edges.back();

            cur = chance;
            searchStack.push_back(cur);
            continue;  // staged outcome consumed at the top of the loop next iteration
        }

        // Deterministic action: the rng was never read, so undo the pre-seed, then deduplicate
        // into a decision node. Move `next` in: it is consumed only if a new node is created
        // (see getOrCreateNode), so the two reads of `next` below are safe -- each is reached
        // only when the node already existed and `next` was left intact.
        next.rng = preActionRng;
        Node* child = getOrCreateNode(std::move(next));
        if (!lastNodeWasCreated) ++stats.detTranspositions;
        edge.node = child;

        if (onPath(child)) {
            // Existing on-path node (a brand-new node is never on the path) -> next intact.
            // Deterministic action cycled back to an on-path node. Roll out instead.
            BattleContext &rollout = (rolloutScratch = next);  // next == child->state
            rolloutToEnd(rollout, actionStack);
            updateFromPlayout(searchStack, actionStack, rollout);
            return;
        }

        searchStack.push_back(child);

        if (child->simulationCount == 0) {
            // Brand-new node (only new nodes have simulationCount 0 here) -> next was moved into
            // child->state, which now holds exactly what next did (rng included).
            BattleContext &rollout = (rolloutScratch = child->state);
            rolloutToEnd(rollout, actionStack);
            updateFromPlayout(searchStack, actionStack, rollout);
            return;
        }
        cur = child;  // transposed into an existing node: keep selecting downward
    }
}

void search::BattleSearcher::updateFromPlayout(const std::vector<Node *> &stack, const std::vector<Action> &actionStack, const BattleContext &endState) {
    // Every simulation ends here exactly once: record how deep it got in-tree and how many
    // chance nodes sat on its path (budget-fragmentation telemetry).
    stats.depthSum += static_cast<std::int64_t>(stack.size());
    for (const auto *n : stack) {
        if (n->isRandomNode) {
            ++stats.chanceDepthSum;
        }
    }
    const auto evaluation = evaluateEndState(endState);

    for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
        auto &node = *(*it);
        ++node.simulationCount;
        node.evaluationSum += evaluation;
    }
}

bool search::BattleSearcher::isTerminalState(const BattleContext &bc) const { // maybe can optimize by making this evaluate directly if score cannot possibly be higher than best
    return bc.outcome != Outcome::UNDECIDED;
}

double search::BattleSearcher::evaluateEdge(const search::BattleSearcher::Node &parent, int edgeIdx, double logParentVisits) {

    const auto &edge = parent.edges[edgeIdx];

    // Explore actions not yet taken from THIS parent first.
    if (edge.visitCount == 0 || edge.node == nullptr) {
        return std::numeric_limits<double>::infinity();
    }

    // Q-value uses the (possibly transposition-shared) child state estimate, but the exploration
    // term uses this edge's own visit count rather than the child's simulationCount. With node
    // sharing the two diverge, and using edge visits keeps UCB well-formed (Childs et al. 2008).
    const double qualityValue = edge.node->evaluationSum / edge.node->simulationCount;
    const double exploration = edge.node->isRandomNode ? explorationParameterChance
                                                       : explorationParameter;
    const double explorationValue = exploration *
            std::sqrt(logParentVisits / (edge.visitCount + 1));

    return qualityValue + explorationValue;
}

int search::BattleSearcher::selectBestEdgeToSearch(const search::BattleSearcher::Node &cur) {
    if (cur.edges.size() == 1) {
        return 0;
    }

    // log(parent visits) is the same for every edge, so compute it once for the whole comparison.
    const double logParentVisits = std::log(cur.simulationCount + 1);

    auto bestEdge = 0;
    auto bestEdgeValue = evaluateEdge(cur, bestEdge, logParentVisits);

    //std::cout << "  edges: " << bestEdgeValue;
    for (int i = 1; i < cur.edges.size(); ++i) {
        const auto value = evaluateEdge(cur, i, logParentVisits);
        //std::cout << ", " << value;
        if (value > bestEdgeValue) {
            bestEdge = i;
            bestEdgeValue = value;
        }
    }
    //std::cout << "\n";
    return bestEdge;
}

// STS_MAX_ROLLOUT_TURNS: if >0, cut the rollout short after that many turns from the start state
// and let evaluateEndState's non-terminal branch heuristic the result. Default 0 = original
// roll-to-terminal behavior.
static int rolloutMaxTurnsEnv() {
    static const int v = [](){
        if (const char *s = std::getenv("STS_MAX_ROLLOUT_TURNS")) return std::atoi(s);
        return 0;
    }();
    return v;
}

void search::BattleSearcher::rolloutToEnd(BattleContext &bc, std::vector<Action> &actionStack) {
    const int maxTurns = rolloutMaxTurnsEnv();
    const int startTurn = bc.turn;
    while (!isTerminalState(bc)) {
        if (maxTurns > 0 && bc.turn - startTurn >= maxTurns) break;
        ++simulationIdx;
        Action action;
        switch (bc.inputState) {
            case InputState::PLAYER_NORMAL:
                // Resolve any hidden (Runic Dome) intent with a sample roll before the rollout agent
                // decides, so it sees the real incoming damage and blocks sensibly. Each rollout's
                // rng samples a different move, so leaf values average to the true EV -- far better
                // than the flat 5*act guess, which made the agent under-block and die, leaving
                // card-play nodes underestimated and unexplored. Isolated to this rollout's scratch
                // bc; the tree still models the hidden intent honestly via its END_TURN chance node.
                for (int i = 0; i < bc.monsters.monsterCount; ++i) {
                    auto &m = bc.monsters.arr[i];
                    if (m.pendingMoveRolls > 0 && !m.isDeadOrEscaped()) {
                        m.materializePendingMove(bc);
                    }
                }
                // Rollout potion policy (STS_ROLLOUT_POTION_MODE) gets first refusal; default
                // mode never drinks, leaving potion plays to the search tree.
                if (!rolloutAgent.choosePotionAction(bc, action)) {
                    action = rolloutAgent.chooseBattleCardPlay(bc);
                }
                break;

            case InputState::CARD_SELECT:
                action = rolloutAgent.chooseBattleCardSelect(bc);
                break;

            default:
#ifdef sts_asserts
                std::cerr << "rolloutToEnd: invalid input state: " << static_cast<int>(bc.inputState) << std::endl;
                assert(false);
#endif
                break;
        }

        actionStack.push_back(action);
        action.execute(bc);
    }
}

search::BattleSearcher::Edge* search::BattleSearcher::selectChanceOutcome(search::BattleSearcher::Node &chance) {
#ifdef sts_asserts
    assert(chance.isRandomNode);
    assert(chance.parent != nullptr);
#endif
    const int wcat = chanceCat(chance.stochasticAction);
    // Double Progressive Widening. Below the cap we draw a fresh i.i.d. outcome; at the cap we
    // reuse an existing one. This bounds the branching of high-entropy events so their children
    // accumulate visits and the tree deepens, while the visit-weighted average over outcomes
    // remains an unbiased estimate of the chance node's expectation.
    const std::int64_t n = chance.simulationCount;
    const double wc = wcat == 0 ? endTurnWideningC : chanceWideningC;
    const double wa = wcat == 0 ? endTurnWideningAlpha : chanceWideningAlpha;
    const int cap = std::max(1, static_cast<int>(
            std::ceil(wc * std::pow(static_cast<double>(n + 1), wa))));

    if (static_cast<int>(chance.edges.size()) < cap) {
        // Widen: reseed from the canonical pre-action state, re-execute, dedup by state.
        // Sequential N gives i.i.d. samples because Random hashes its seed (murmurHash3).
        const std::uint64_t N = chance.outcomesGenerated++;
        BattleContext &out = (widenScratch = chance.parent->state);
        out.rng = Random(chance.randomnessBase + N);
        chance.stochasticAction.execute(out);

        ++stats.chanceOutcomesSampled;
        ++stats.wSampled[wcat];
        Node* child = getOrCreateNode(std::move(out));  // out unused afterward; safe to move
        for (auto &e : chance.edges) {
            if (e.node == child) {
                ++stats.chanceSiblingReuse;
                ++stats.wSibReuse[wcat];
                return &e;  // resampled an outcome we already have
            }
        }
        if (!lastNodeWasCreated) ++stats.chanceTranspositions;

        Edge e;
        e.action = Action{};
        e.node = child;
        e.rngAdvanceSteps = static_cast<int>(N);
        chance.edges.push_back(std::move(e));
        return &chance.edges.back();
    }

    // Capped: re-select an existing outcome proportional to its visit count, so the realized
    // descent frequencies keep tracking the true outcome probabilities.
    ++stats.wCapped[wcat];
    std::int64_t totalVisits = 0;
    for (const auto &e : chance.edges) {
        totalVisits += e.visitCount;
    }
    if (totalVisits <= 0) {
        return &chance.edges[0];
    }
    std::uniform_int_distribution<std::int64_t> dist(0, totalVisits - 1);
    std::int64_t r = dist(randGen);
    for (auto &e : chance.edges) {
        r -= e.visitCount;
        if (r < 0) {
            return &e;
        }
    }
    return &chance.edges.back();
}

void search::BattleSearcher::enumerateActionsForNode(search::BattleSearcher::Node &node,
                                                               const BattleContext &bc) {
    switch (bc.inputState) {
        case InputState::PLAYER_NORMAL:
            enumerateCardActions(node, bc);
            enumeratePotionActions(node, bc);
            node.edges.push_back({Action(ActionType::END_TURN)});
            break;

        case InputState::CARD_SELECT:
            enumerateCardSelectActions(node, bc);
            break;

        default:
#ifdef sts_asserts
            std::cerr << "enumerateActionsForNode: invalid input state: " << static_cast<int>(bc.inputState) << std::endl;
            assert(false);
#endif
            break;
    }

#ifdef sts_print_debug
    std::cout << "{ (" << node.edges.size() << ") ";
    for (int i = 0; i < node.edges.size(); ++i) {
        node.edges[i].action.printDesc(std::cout, bc) << ", ";
    }
    std::cout << " }" << std::endl;
#endif
}

void search::BattleSearcher::enumerateCardActions(search::BattleSearcher::Node &node,
                                                            const BattleContext &bc) {
    if (!bc.isCardPlayAllowed()) {
        return;
    }

    fixed_list<std::pair<int,int>, 10> playableHandIdxs;
    for (int handIdx = 0; handIdx < bc.cards.cardsInHand; ++handIdx) {
        const auto &c = bc.cards.hand[handIdx];
        if (!c.canUseOnAnyTarget(bc)) {
            continue;
        }

        bool isUniqueAction = true;

        if (handIdx > 0) {
            const auto &lastCard = bc.cards.hand[handIdx-1];

            bool isEqualToLastCard = c.id == lastCard.id &&
                    c.getUpgradeCount() == lastCard.getUpgradeCount() &&
                    // both should be less than deck size c.uniqueId < bc.cards.deck
                    c.costForTurn == lastCard.costForTurn &&
                    c.cost == lastCard.cost &&
                    c.freeToPlayOnce == lastCard.freeToPlayOnce &&
                    c.specialData == lastCard.specialData;

            if (isEqualToLastCard) {
                isUniqueAction = false;
            }
        }

        if (isUniqueAction) {
            playableHandIdxs.push_back( {handIdx, search::Expert::getPlayOrdering(c.getId())} );
        }
    }

    std::sort(playableHandIdxs.begin(), playableHandIdxs.end(), [](auto a, auto b) { return a.second < b.second; });

    for (auto pair : playableHandIdxs) {
        const auto handIdx = pair.first;
        const auto &c = bc.cards.hand[handIdx];

        if (c.requiresTarget()) {
            for (int tIdx = bc.monsters.monsterCount-1; tIdx >= 0; --tIdx) {
                if (!bc.monsters.arr[tIdx].isTargetable()) {
                    continue;
                }
                node.edges.push_back({Action(ActionType::CARD, handIdx, tIdx)});
            }
        } else {
            node.edges.push_back({Action(ActionType::CARD, handIdx)});
        }
    }

}

void search::BattleSearcher::enumeratePotionActions(search::BattleSearcher::Node &node,
                                                              const BattleContext &bc) {

    const auto hasValidTarget = bc.monsters.getTargetableCount() > 0;

    int foundPotions = 0;
    for (int pIdx = 0; pIdx < bc.potionCapacity; ++pIdx) {

        const auto p = bc.potions[pIdx];
        if (p == Potion::EMPTY_POTION_SLOT) {
            continue;
        }
        ++foundPotions;

        // not enumerating the discard of a potion if it can be used
        if (p == Potion::FAIRY_POTION) {
            node.edges.push_back({Action(ActionType::POTION, pIdx, -1)});
            continue;
        }

        // Offer only Smoke Bomb's discard when it can't flee -- the search otherwise over-picks
        // the high-value escape and the live game rejects the "use" command.
        if (p == Potion::SMOKE_BOMB && bc.smokeBombEscapeBlocked()) {
            node.edges.push_back({Action(ActionType::POTION, pIdx, -1)});
            continue;
        }

        if (!potionRequiresTarget(p)) {
            node.edges.push_back({Action(ActionType::POTION, pIdx)});
            continue;
        }

        // potion requires target
        if (!hasValidTarget) {
            node.edges.push_back({Action(ActionType::POTION, pIdx, -1)});
            continue;
        }

        // there is a valid target
        for (int tIdx = 0; tIdx < bc.monsters.monsterCount; ++tIdx) {
            if (bc.monsters.arr[tIdx].isTargetable()) {
                node.edges.push_back({Action(ActionType::POTION, pIdx, tIdx)});
            }
        }
    }
}

void search::BattleSearcher::enumerateCardSelectActions(search::BattleSearcher::Node &node,
                                                                  const BattleContext &bc) {
    // Action::enumerateCardSelectActions is the single source of truth for which selects a
    // card-select screen offers; the search just wraps each one in an edge.
    for (const auto &a : Action::enumerateCardSelectActions(bc)) {
        node.edges.push_back({a});
    }
}

double getNonMinionMonsterCurHpRatio(const BattleContext &bc) {
    int curHpTotal = 0;
    int maxHpTotal = 0;

    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        const auto &m = bc.monsters.arr[i];
        if (!m.hasStatus<MS::MINION>() && m.id != sts::MonsterId::INVALID) {
            curHpTotal += m.curHp;
            maxHpTotal += m.maxHp;
        }
    }

    if (curHpTotal == 0 || maxHpTotal == 0) {
        return 0;
    }

    return (double)curHpTotal / maxHpTotal;
}

// The gold the player effectively holds in `bc`: their pocket plus any stolen gold a NOT-escaped
// Looter/Mugger still carries (exitBattle refunds it whether the thief is dead or merely present
// at the win). Gold on an ESCAPED thief is gone and excluded.
static double effectiveGold(const BattleContext &bc) {
    double gold = bc.player.gold;
    if (bc.requiresStolenGoldCheck()) {
        for (int i = 0; i < bc.monsters.monsterCount; ++i) {
            const auto &m = bc.monsters.arr[i];
            const bool thief = m.id == MonsterId::LOOTER || m.id == MonsterId::MUGGER;
            const bool escaped = m.curHp > 0 && (m.moveHistory[0] == MonsterMoveId::LOOTER_ESCAPE ||
                                                 m.moveHistory[0] == MonsterMoveId::MUGGER_ESCAPE);
            if (thief && !escaped) {
                gold += m.miscInfo;
            }
        }
    }
    return gold;
}

double search::BattleSearcher::evaluateEndState(const BattleContext &bc) const {
    const double potionScore = bc.potionCount * evalWeights.potionWeight;
    // Credit held "cheat death" effects at the HP they would actually restore, so the search keeps
    // them rather than playing into lethal to burn them for a marginal heal (the search otherwise
    // prefers TRIGGERING the save -- revive, then win at low HP -- over ending the fight intact).
    // Mirror Player::wouldDie exactly: Fairy in a Bottle fires first and heals 30% max HP (60% with
    // Sacred Bark); Lizard Tail fires next and heals to 50% max HP; Mark of the Bloom disables all
    // healing so neither saves you. Holding both is two independent lives, so the values add.
    // Scaled at HP's ~1:1 weight (matches postBattleHealedHp / the loss-branch hp ratio).
    double deathSaveValue = 0.0;
    if (!bc.player.hasRelic<RelicId::MARK_OF_THE_BLOOM>()) {
        bool hasFairy = false;
        for (int i = 0; i < bc.potionCapacity; ++i) {
            if (bc.potions[i] == Potion::FAIRY_POTION) { hasFairy = true; break; }
        }
        if (hasFairy) {
            deathSaveValue += bc.player.maxHp * (bc.player.hasRelic<RelicId::SACRED_BARK>() ? 0.6 : 0.3);
        }
        if (bc.player.hasRelic<RelicId::LIZARD_TAIL>()) {
            deathSaveValue += 0.5 * bc.player.maxHp;
        }
    }

    if (bc.outcome == Outcome::PLAYER_VICTORY) {
        // postBattleHealedHp: HP after a boss victory reflects the act-transition heal, so the
        // search doesn't value preserving HP that the game is about to restore anyway.
        double score = evalWeights.winBonus + bc.postBattleHealedHp() + potionScore - (bc.turn * evalWeights.victoryTurnPenalty) + deathSaveValue;
        if (evalWeights.goldWeight != 0) {
            // Effective gold delta vs the search root: values gold gained in combat (Hand of
            // Greed) and charges gold lost to an escaped thief at the same weight. Gold held by a
            // NON-escaped thief still counts as the player's -- kills refund it at exitBattle
            // (steal-then-kill nets the same as no steal); an escape drops it from the sum, so the
            // steal shows up as a plain negative delta. Root-baselined like maxHpWeight so the
            // term is a pure delta and cannot distort the victory/loss tradeoff.
            score += evalWeights.goldWeight * (effectiveGold(bc) - effectiveGold(*rootState));
        }
        if (evalWeights.maxHpWeight != 0) {
            // Max HP gained during the battle (Feed, Darkstone Periapt); baselined at the search
            // root so the term is a pure delta and cannot distort the victory/loss tradeoff.
            score += evalWeights.maxHpWeight * (bc.player.maxHp - rootState->player.maxHp);
        }
        if (evalWeights.parasitePenalty != 0) {
            // Mirrors exitBattle: an implanted Writhing Mass adds a Parasite to the deck unless
            // a charged Omamori absorbs it.
            const auto &wm = bc.monsters.arr[0];
            if (wm.id == MonsterId::WRITHING_MASS && wm.miscInfo
                && (!bc.player.hasRelic<RelicId::OMAMORI>()
                    || bc.gameContext->relics.getRelicValue(RelicId::OMAMORI) == 0)) {
                score -= evalWeights.parasitePenalty;
            }
        }
        return score;
    } else {
        const bool couldHaveSpikers = bc.encounter == MonsterEncounter::THREE_SHAPES || bc.encounter == MonsterEncounter::FOUR_SHAPES;
        const double energyPenalty = bc.energyWasted * -evalWeights.energyWasteWeight * (couldHaveSpikers ? 0 : 1);
        const double drawBonus = bc.cardsDrawn * evalWeights.drawWeight;
        const double aliveScore = bc.monsters.monstersAlive * -evalWeights.aliveWeight;
        // Survival credit rewards dying later in a genuinely lost fight -- but the engine's
        // infinite-fight safety valve (executeActions: turn > 100 => PLAYER_LOSS) would earn
        // 1.5/turn * 100+, outscoring any victory, so the search deliberately stalls winnable
        // fights to the cap (observed live: a 103-turn Gremlin Gang). Zero the credit for
        // cap losses; real deaths always come far earlier and keep it.
        const double survivalScore = bc.turn > 100 ? 0 : bc.turn * evalWeights.turnSurvivalWeight;

        return (1 - getNonMinionMonsterCurHpRatio(bc)) * evalWeights.monsterDamageWeight + aliveScore + energyPenalty + drawBonus + potionScore / 2 + survivalScore + deathSaveValue;
    }
}

search::Action search::BattleSearcher::getBestAction() const {
    if (root->edges.empty()) {
        throw std::runtime_error("BattleSearcher::getBestAction() called with no available actions");
    }

    int bestEdgeIdx = 0;
    std::int32_t maxVisits = root->edges[0].visitCount;

    for (int i = 1; i < root->edges.size(); ++i) {
        if (root->edges[i].visitCount > maxVisits) {
            maxVisits = root->edges[i].visitCount;
            bestEdgeIdx = i;
        }
    }

    return root->edges[bestEdgeIdx].action;
}

// (edge, state-to-describe-the-action-in). Reads stored node state directly -- no replay.
typedef std::pair<const search::BattleSearcher::Edge*, const BattleContext*> EdgeInfo;

// Collect the edges of every node at the given depth (root == depth 1). An edge's action is
// described in its parent decision node's state; chance-node outcome edges carry dummy actions
// and borrow the chance node's parent state. Shared (transposed) nodes are printed once.
std::vector<EdgeInfo> getEdgesForLayer(const search::BattleSearcher &s, int layerNum) {
    if (layerNum <= 0) {
        return {};
    }

    std::vector<EdgeInfo> layerEdges;
    std::unordered_set<const search::BattleSearcher::Node*> seen;
    std::vector<std::pair<const search::BattleSearcher::Node*, int>> stack { {s.root, 1} };

    while (!stack.empty()) {
        const auto [node, depth] = stack.back();
        stack.pop_back();
        if (node == nullptr || !seen.insert(node).second) {
            continue;
        }

        const BattleContext* describeState =
                node->isRandomNode ? (node->parent ? &node->parent->state : nullptr)
                                   : &node->state;

        if (depth == layerNum) {
            for (const auto &edge : node->edges) {
                layerEdges.emplace_back(&edge, describeState);
            }
            continue;
        }

        for (const auto &edge : node->edges) {
            if (edge.node != nullptr) {
                stack.push_back({edge.node, depth + 1});
            }
        }
    }

    return layerEdges;
}

void search::BattleSearcher::printSearchTree(std::ostream &os, int levels) {
    for (int depth = 1; depth <= levels; ++depth) {
        for (const auto &x : getEdgesForLayer(*this, depth)) {
            os << "(" << (x.first->node ? x.first->node->simulationCount : 0) << ")";
            if (x.second != nullptr) {
                x.first->action.printDesc(os, *x.second) << "\t";
            }
        }
        os << '\n';
    }
}

void search::BattleSearcher::printSearchStack(std::ostream &os, bool skipLast) {
    for (int i = 0; i < actionStack.size(); ++i) {
        const auto &a = actionStack[i];
        os << std::hex << a.bits << '\n';
    }

    os.flush();

//    BattleContext curBc = *rootState;
//    os << "explorationParameter: " << explorationParameter << '\n';
//    os << "bestActionValue: " << bestActionValue << '\n';
//    os << "minActionValue: " << minActionValue << '\n';
//    os << "outcomePlayerHp: " << outcomePlayerHp << '\n';
//    os << "root node:\n";
//    os << curBc << "\n";
//
//    for (int i = 0; i < actionStack.size(); ++i) {
//        if (i < searchStack.size()) {
//            const auto &n = searchStack[i];
//            os << i << " nodeSearched: " << n->simulationCount << " { ";
//            for (const auto &edge : n->edges) {
//                os << "(" << edge.node.simulationCount << ")";
//                edge.action.printDesc(os, curBc) << " ";
//            }
//            os << "}\n";
//        }
//
//        const auto &a = actionStack[i];
//        os << i << " actionTaken: ";
//        a.printDesc(os, curBc) << '\n';
//
//        if (skipLast && (i + 1 >= actionStack.size())) {
//            break;
//        }
//
//        a.execute(curBc);
//        os << curBc << '\n';
//    }
//
//    os.flush();
}
