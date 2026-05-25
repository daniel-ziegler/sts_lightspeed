//
// Created by keega on 9/18/2021.
//

#include "sim/search/BattleSearcher.h"
#include "sim/search/ExpertKnowledge.h"

#include <algorithm>
#include <utility>
#include <string>
#include <memory>

using namespace sts;

std::int64_t simulationIdx = 0; // for debugging

namespace sts::search {
    thread_local search::BattleSearcher *g_debug_scum_search;

    // FNV-1a hash constants for 64-bit
    constexpr std::uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
    constexpr std::uint64_t FNV_PRIME = 1099511628211ULL;

    // Hash a value into accumulator using FNV-1a (better distribution than simple XOR)
    inline void hash_combine_fnv(std::uint64_t& hash, std::uint64_t value) {
        // Split value into bytes for better mixing
        for (int i = 0; i < 8; ++i) {
            hash ^= (value & 0xFF);
            hash *= FNV_PRIME;
            value >>= 8;
        }
    }

    inline void hash_combine_fnv(std::uint64_t& hash, std::uint32_t value) {
        for (int i = 0; i < 4; ++i) {
            hash ^= (value & 0xFF);
            hash *= FNV_PRIME;
            value >>= 8;
        }
    }

    inline void hash_combine_fnv(std::uint64_t& hash, int value) {
        hash_combine_fnv(hash, static_cast<std::uint32_t>(value));
    }

    inline void hash_combine_fnv(std::uint64_t& hash, bool value) {
        hash ^= (value ? 1 : 0);
        hash *= FNV_PRIME;
    }

    // Hash BattleContext state for graph search deduplication
    // Only hashes observable game state, not internal RNG or debug counters
    std::size_t hashBattleState(const BattleContext& bc) {
        std::uint64_t hash = FNV_OFFSET_BASIS;

        // Hash turn and outcome
        hash_combine_fnv(hash, bc.turn);
        hash_combine_fnv(hash, static_cast<int>(bc.outcome));
        hash_combine_fnv(hash, static_cast<int>(bc.inputState));

        // Hash player state
        const auto& p = bc.player;
        hash_combine_fnv(hash, p.curHp);
        hash_combine_fnv(hash, p.maxHp);
        hash_combine_fnv(hash, p.block);
        hash_combine_fnv(hash, p.energy);
        hash_combine_fnv(hash, p.cardsPlayedThisTurn);
        hash_combine_fnv(hash, p.statusBits0);
        hash_combine_fnv(hash, p.statusBits1);
        hash_combine_fnv(hash, p.strength);
        hash_combine_fnv(hash, p.dexterity);

        // Hash monsters
        for (int i = 0; i < bc.monsters.monsterCount; ++i) {
            const auto& m = bc.monsters.arr[i];
            hash_combine_fnv(hash, static_cast<int>(m.id));
            hash_combine_fnv(hash, m.curHp);
            hash_combine_fnv(hash, m.maxHp);
            hash_combine_fnv(hash, m.block);
            hash_combine_fnv(hash, m.statusBits);
            hash_combine_fnv(hash, static_cast<int>(m.moveHistory[0]));
        }

        // Hash cards in hand (INCLUDING POSITION to prevent invalid deduplication)
        hash_combine_fnv(hash, bc.cards.cardsInHand);
        for (int i = 0; i < bc.cards.cardsInHand; ++i) {
            const auto& c = bc.cards.hand[i];
            hash_combine_fnv(hash, i); // Hash position to preserve ordering
            hash_combine_fnv(hash, static_cast<int>(c.id));
            hash_combine_fnv(hash, c.cost);
            hash_combine_fnv(hash, c.costForTurn);
            hash_combine_fnv(hash, c.upgraded);
        }

        // Hash draw pile (including top cards with position for ordering)
        hash_combine_fnv(hash, static_cast<int>(bc.cards.drawPile.size()));
        int drawToHash = std::min(5, static_cast<int>(bc.cards.drawPile.size()));
        for (int i = 0; i < drawToHash; ++i) {
            hash_combine_fnv(hash, i); // Position in draw pile
            hash_combine_fnv(hash, static_cast<int>(bc.cards.drawPile[i].id));
        }

        // Hash discard pile (including position for ordering)
        hash_combine_fnv(hash, static_cast<int>(bc.cards.discardPile.size()));
        int discardIdx = 0;
        for (const auto& c : bc.cards.discardPile) {
            hash_combine_fnv(hash, discardIdx++); // Position in discard
            hash_combine_fnv(hash, static_cast<int>(c.id));
        }

        // Hash exhaust pile
        hash_combine_fnv(hash, static_cast<int>(bc.cards.exhaustPile.size()));

        // Hash potions
        hash_combine_fnv(hash, bc.potionCount);
        for (int i = 0; i < bc.potionCount; ++i) {
            hash_combine_fnv(hash, static_cast<int>(bc.potions[i]));
        }

        return hash;
    }
}



search::BattleSearcher::BattleSearcher(const BattleContext &bc, search::EvalFnc _evalFnc)
    : rootState(new BattleContext(bc)), evalFnc(std::move(_evalFnc)), randGen(bc.seed+bc.floorNum), rolloutAgent(true, bc.seed+bc.floorNum) {
}

search::BattleSearcher::~BattleSearcher() {
    stateToNode.clear();
    allNodes.clear();
}

search::BattleSearcher::Node* search::BattleSearcher::getOrCreateNode(const BattleContext &state) {
    const size_t hash = search::hashBattleState(state);

    // Resolve hash collisions exactly: only reuse a node whose stored state is
    // search-equal. This is what makes transposition safe without a perfect hash.
    auto &bucket = stateToNode[hash];
    for (Node* candidate : bucket) {
        if (candidate->state.equalForSearch(state)) {
            return candidate;
        }
    }

    allNodes.push_back(std::make_unique<Node>());
    Node* newNode = allNodes.back().get();
    newNode->state = state;
    bucket.push_back(newNode);
    return newNode;
}

void search::BattleSearcher::search(int64_t simulations) {
    g_debug_scum_search = this;

    // Fresh search: reset the node pool and root (root.edges hold raw pointers into the pool).
    allNodes.clear();
    stateToNode.clear();
    root = Node();
    root.state = *rootState;

    if (isTerminalState(root.state)) {
        root.evaluationSum = evaluateEndState(root.state);
        root.simulationCount = 1;
        return;
    }

    for (std::int64_t simCount = 0; simCount < simulations; ++simCount) {
        step();
    }
}

void search::BattleSearcher::step() {
    searchStack.clear();
    searchStack.push_back(&root);
    actionStack.clear();

    Node* cur = &root;
    int depth = 0;
    while (true) {
        // Safety valve. With proper chance handling and a monotonic turn counter this
        // should never trigger; a cycle guard replaces it in a later commit.
        if (++depth > 5000) {
            return;
        }

        if (cur->isRandomNode) {
            // Chance node: resolve one outcome (sampled/selected), descend into it.
            Edge* outcomeEdge = selectChanceOutcome(*cur);
            ++outcomeEdge->visitCount;
            Node* child = outcomeEdge->node;
            searchStack.push_back(child);

            if (child->simulationCount == 0) {
                BattleContext rollout = child->state;
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
            actionStack.push_back(edge.action);
            cur = edge.node;
            searchStack.push_back(cur);
            continue;
        }

        // First traversal of this action: execute on a copy of the current state.
        BattleContext next = cur->state;
        const auto rngCounterBefore = next.rng.counter;
        Random preActionRng = next.rng;
        edge.action.execute(next);
        actionStack.push_back(edge.action);
        const bool rngChanged = next.rng.counter != rngCounterBefore;

        if (rngChanged) {
            // Stochastic action: create a chance node that sources its pre-action state
            // from `cur` and resolves outcomes via Random(randomnessBase + N).
            allNodes.push_back(std::make_unique<Node>());
            Node* chance = allNodes.back().get();
            chance->isRandomNode = true;
            chance->stochasticAction = edge.action;
            chance->parent = cur;
            chance->randomnessBase = static_cast<std::uint64_t>(preActionRng.randomLong());
            edge.node = chance;
            cur = chance;
            searchStack.push_back(cur);
            continue;  // outcome resolved at the top of the loop next iteration
        }

        // Deterministic action: deduplicate into a decision node.
        Node* child = getOrCreateNode(next);
        edge.node = child;
        searchStack.push_back(child);

        if (child->simulationCount == 0) {
            BattleContext rollout = next;  // next == child->state
            rolloutToEnd(rollout, actionStack);
            updateFromPlayout(searchStack, actionStack, rollout);
            return;
        }
        cur = child;  // transposed into an existing node: keep selecting downward
    }
}

void search::BattleSearcher::updateFromPlayout(const std::vector<Node *> &stack, const std::vector<Action> &actionStack, const BattleContext &endState) {
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

double search::BattleSearcher::evaluateEdge(const search::BattleSearcher::Node &parent, int edgeIdx) {

    const auto &edge = parent.edges[edgeIdx];

    // Unvisited edges (nullptr or zero visits) should be explored first
    if (edge.node == nullptr || edge.node->simulationCount == 0) {
        return std::numeric_limits<double>::infinity();
    }

    double qualityValue = edge.node->evaluationSum / edge.node->simulationCount;

    double explorationValue = explorationParameter *
            std::sqrt(std::log(parent.simulationCount+1) / (edge.node->simulationCount+1));

    return qualityValue + explorationValue;
}

int search::BattleSearcher::selectBestEdgeToSearch(const search::BattleSearcher::Node &cur) {
    if (cur.edges.size() == 1) {
        return 0;
    }

    auto bestEdge = 0;
    auto bestEdgeValue = evaluateEdge(cur, bestEdge);

    //std::cout << "  edges: " << bestEdgeValue;
    for (int i = 1; i < cur.edges.size(); ++i) {
        const auto value = evaluateEdge(cur, i);
        //std::cout << ", " << value;
        if (value > bestEdgeValue) {
            bestEdge = i;
            bestEdgeValue = value;
        }
    }
    //std::cout << "\n";
    return bestEdge;
}

int search::BattleSearcher::selectFirstActionForLeafNode(const search::BattleSearcher::Node &leafNode) {
    auto dist = std::uniform_int_distribution<int>(0, static_cast<int>(leafNode.edges.size())-1);
    return dist(randGen);
}

void search::BattleSearcher::rolloutToEnd(BattleContext &bc, std::vector<Action> &actionStack) {
    while (!isTerminalState(bc)) {
        ++simulationIdx;
        Action action;
        switch (bc.inputState) {
            case InputState::PLAYER_NORMAL:
                action = rolloutAgent.chooseBattleCardPlay(bc);
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
    // Double Progressive Widening. Below the cap we draw a fresh i.i.d. outcome; at the cap we
    // reuse an existing one. This bounds the branching of high-entropy events so their children
    // accumulate visits and the tree deepens, while the visit-weighted average over outcomes
    // remains an unbiased estimate of the chance node's expectation.
    const std::int64_t n = chance.simulationCount;
    const int cap = std::max(1, static_cast<int>(
            std::ceil(chanceWideningC * std::pow(static_cast<double>(n + 1), chanceWideningAlpha))));

    if (static_cast<int>(chance.edges.size()) < cap) {
        // Widen: reseed from the canonical pre-action state, re-execute, dedup by state.
        // Sequential N gives i.i.d. samples because Random hashes its seed (murmurHash3).
        const std::uint64_t N = chance.outcomesGenerated++;
        BattleContext out = chance.parent->state;
        out.rng = Random(chance.randomnessBase + N);
        chance.stochasticAction.execute(out);

        Node* child = getOrCreateNode(out);
        for (auto &e : chance.edges) {
            if (e.node == child) {
                return &e;  // resampled an outcome we already have
            }
        }

        Edge e;
        e.action = Action{};
        e.node = child;
        e.rngAdvanceSteps = static_cast<int>(N);
        chance.edges.push_back(std::move(e));
        return &chance.edges.back();
    }

    // Capped: re-select an existing outcome proportional to its visit count, so the realized
    // descent frequencies keep tracking the true outcome probabilities.
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

template <typename ForwardIt>
void setupCardOptionsHelper(search::BattleSearcher::Node &node, const ForwardIt begin, const ForwardIt end, const std::function<bool(const CardInstance &)> &p= nullptr) {
    for (int i = 0; begin+i != end; ++i) {
        const auto &c = begin[i];
        if (!p || (p(c))) {
            node.edges.push_back(
                    {search::Action(search::ActionType::SINGLE_CARD_SELECT, i)}
                );
        }
    }
}

void search::BattleSearcher::enumerateCardSelectActions(search::BattleSearcher::Node &node,
                                                                  const BattleContext &bc) {

    switch (bc.cardSelectInfo.cardSelectTask) {
        case CardSelectTask::ARMAMENTS:
            setupCardOptionsHelper( node, bc.cards.hand.begin(), bc.cards.hand.begin() + bc.cards.cardsInHand,
                                    [] (const CardInstance &c) { return c.canUpgrade(); });
            break;

        case CardSelectTask::CODEX:
            for (int i = 0; i < 4; ++i) { // i -> 3 action means skip
                node.edges.push_back({Action(search::ActionType::SINGLE_CARD_SELECT, i)});
            }
            break;

        case CardSelectTask::DISCOVERY:
            for (int i = 0; i < 3; ++i) {
                node.edges.push_back({Action(search::ActionType::SINGLE_CARD_SELECT, i)});
            }
            break;

        case CardSelectTask::DUAL_WIELD:
            setupCardOptionsHelper( node, bc.cards.hand.begin(), bc.cards.hand.begin() + bc.cards.cardsInHand,
                                    [] (const CardInstance &c) {
                                        return c.getType() == CardType::POWER || c.getType() == CardType::ATTACK;
                                    });
            break;

        case CardSelectTask::EXHUME:
            setupCardOptionsHelper(node, bc.cards.exhaustPile.begin(), bc.cards.exhaustPile.end(),
                                   [](const auto &c) { return c.getId() != CardId::EXHUME; });
            break;

        case CardSelectTask::EXHAUST_ONE:
            setupCardOptionsHelper(node, bc.cards.hand.begin(), bc.cards.hand.begin() + bc.cards.cardsInHand);
            break;

        case CardSelectTask::FORETHOUGHT:
        case CardSelectTask::WARCRY:
            setupCardOptionsHelper(node, bc.cards.hand.begin(), bc.cards.hand.begin() + bc.cards.cardsInHand);
            break;

        case CardSelectTask::HEADBUTT:
        case CardSelectTask::LIQUID_MEMORIES_POTION:
            setupCardOptionsHelper(node, bc.cards.discardPile.begin(), bc.cards.discardPile.end());
            break;

        case CardSelectTask::SECRET_TECHNIQUE:
            setupCardOptionsHelper(node, bc.cards.drawPile.begin(), bc.cards.drawPile.end(),
                                    [] (const CardInstance &c) {
                                        return c.getType() == CardType::SKILL;
                                    });
            break;

        case CardSelectTask::SECRET_WEAPON:
            setupCardOptionsHelper(node, bc.cards.drawPile.begin(), bc.cards.drawPile.end(),
                                    [] (const CardInstance &c) {
                                        return c.getType() == CardType::ATTACK;
                                    });
            break;

        case CardSelectTask::EXHAUST_MANY:
        case CardSelectTask::GAMBLE:
            // just dont deal with this right now
            node.edges.push_back({search::Action(search::ActionType::MULTI_CARD_SELECT, 0)});
            break;

        default:
#ifdef sts_asserts
            assert(false);
#endif
            break;
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

double search::BattleSearcher::evaluateEndState(const BattleContext &bc) {
    double potionScore = bc.potionCount * 12;

    if (bc.outcome == Outcome::PLAYER_VICTORY) {
        double score = 100 + bc.player.curHp + potionScore - (bc.turn * 0.01);
        // std::cout << "Victory! Turn " << bc.turn << " " << bc.player.curHp << "/" << bc.player.maxHp << "hp " << bc.potionCount << "pots: " << score << "\n";
        return score;
    } else {
//        double statusScore =
//                (bc.player.getStatus<PS::STRENGTH>() * .5);
        const bool couldHaveSpikers = bc.encounter == MonsterEncounter::THREE_SHAPES || bc.encounter == MonsterEncounter::FOUR_SHAPES;
        double energyPenalty = bc.energyWasted * -0.2 * (couldHaveSpikers ? 0 : 1);
        double drawBonus = bc.cardsDrawn * 0.03;
        double aliveScore = bc.monsters.monstersAlive*-1;

        return (1-getNonMinionMonsterCurHpRatio(bc))*10 + aliveScore + energyPenalty + drawBonus + potionScore / 2 + (bc.turn * .2);
    }
}

search::Action search::BattleSearcher::getBestAction() const {
    if (root.edges.empty()) {
        throw std::runtime_error("BattleSearcher::getBestAction() called with no available actions");
    }

    int bestEdgeIdx = 0;
    std::int64_t maxVisits = root.edges[0].node ? root.edges[0].node->simulationCount : 0;

    for (int i = 1; i < root.edges.size(); ++i) {
        std::int64_t visits = root.edges[i].node ? root.edges[i].node->simulationCount : 0;
        if (visits > maxVisits) {
            maxVisits = visits;
            bestEdgeIdx = i;
        }
    }

    return root.edges[bestEdgeIdx].action;
}

struct LayerStruct {
    const search::BattleSearcher::Node *node;
    BattleContext *bc;
    int edgeIdx;
};

typedef std::pair<search::BattleSearcher::Edge, std::unique_ptr<const BattleContext>> EdgeInfo;

std::vector<EdgeInfo> getEdgesForLayer(const search::BattleSearcher &s, int layerNum) {
    if (layerNum <= 0) {
        return {};
    }

    std::vector<EdgeInfo> layerEdges;

    std::vector<LayerStruct> curStack { {&s.root, new BattleContext(*s.rootState), 0} };

    while (!curStack.empty()) {
        if (curStack.size() == layerNum) {
            for (const auto &edge : curStack.back().node->edges) {
                layerEdges.emplace_back(edge, std::make_unique<const BattleContext>(*curStack.back().bc));
            }
        }

       // curStack size less than layerNum
       const bool visitedAll = curStack.back().edgeIdx >= curStack.back().node->edges.size();
       if (visitedAll || curStack.size() == layerNum) {
           delete curStack.back().bc;
           curStack.pop_back();
           continue;
       }

        // visit next edge
        auto &nextIdx = curStack.back().edgeIdx;
        const auto *parentNode = curStack.back().node;
        const auto &edgeRef = parentNode->edges[nextIdx];

        BattleContext bc(*curStack.back().bc);
        if (parentNode->isRandomNode) {
            // Parent is random node: constant-time reconstruction using base value + step index
            Random rngCopy = bc.rng;
            const std::uint64_t base = static_cast<std::uint64_t>(rngCopy.randomLong());
            bc.rng = Random(base + static_cast<std::uint64_t>(edgeRef.rngAdvanceSteps));
            const auto &stochAction = parentNode->stochasticAction;
            stochAction.execute(bc);
        } else {
            edgeRef.action.execute(bc);
        }

        curStack.push_back( {edgeRef.node, new BattleContext(bc), 0} );
        ++nextIdx;
    }

    return layerEdges;
}

void search::BattleSearcher::printSearchTree(std::ostream &os, int levels) {
    std::vector<std::vector<EdgeInfo>> layerEdges;
    for (int depth = 1; depth <= levels; ++depth) {
        layerEdges.push_back(getEdgesForLayer(*this, depth));
    }

//    auto maxIt = std::max(layerEdges.begin(), layerEdges.end(), [](auto a, auto b) { return a->size() < b->size(); });
//    if (maxIt == layerEdges.end()) {
//        return;
//    }
//    // maxIt points to something
//    const auto maxSize = maxIt->size();
//    constexpr auto edgeWidth = 30;

    for (int depth = 0; depth < levels; ++depth) {
        for (const auto &x : layerEdges[depth]) {
            os << "(" << (x.first.node ? x.first.node->simulationCount : 0) << ")";
            // We don't print the action for random edges; they reflect RNG branches
            x.first.action.printDesc(os, *x.second) << "\t";
        }
        std::cout << '\n';
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
