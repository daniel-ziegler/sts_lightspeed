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
}



search::BattleSearcher::BattleSearcher(const BattleContext &bc, search::EvalFnc _evalFnc)
    : rootState(new BattleContext(bc)), evalFnc(std::move(_evalFnc)), randGen(bc.seed+bc.floorNum) {
}

search::BattleSearcher::Node::Node(const Node& other)
    : simulationCount(other.simulationCount)
    , evaluationSum(other.evaluationSum)
    , edges(other.edges)
    , isRandomNode(other.isRandomNode)
    , stochasticAction(other.stochasticAction)
    , outcomesGenerated(other.outcomesGenerated)
{}

search::BattleSearcher::Node& search::BattleSearcher::Node::operator=(const Node& other) {
    if (this != &other) {
        simulationCount = other.simulationCount;
        evaluationSum = other.evaluationSum;
        edges = other.edges;
        isRandomNode = other.isRandomNode;
        stochasticAction = other.stochasticAction;
        outcomesGenerated = other.outcomesGenerated;
    }
    return *this;
}

void search::BattleSearcher::search(int64_t simulations) {
    g_debug_scum_search = this;

    if (isTerminalState(*rootState)) {
        auto evaluation = evaluateEndState(*rootState);
        outcomePlayerHp = rootState->player.curHp;
        bestActionSequence = {};

        root.evaluationSum = evaluation;
        root.simulationCount = 1;
    }

    for (std::int64_t simCount = 0; simCount < simulations; ++simCount) {
        step();
    }
}

void search::BattleSearcher::step() {
    searchStack = {&root};
    actionStack.clear();
    BattleContext curState;
    curState = *rootState;

    while (true) {
        auto &curNode = *searchStack.back();

        if (curNode.isRandomNode) {
            expandRandomOutcome(curNode, curState);
            auto &edgeTaken = curNode.edges.back();
            searchStack.push_back(&edgeTaken.node);
            continue;
        }

        if (isTerminalState(curState)) {
            updateFromPlayout(searchStack, actionStack, curState);
            return;
        }

        const bool isLeaf = curNode.edges.empty();
        if (isLeaf) {

            ++simulationIdx;
            enumerateActionsForNode(curNode, curState);
            const auto selectIdx = selectFirstActionForLeafNode(curNode);
            auto &edgeTaken = curNode.edges[selectIdx];

            // Snapshot RNG state before executing action
            const auto rngCounterBefore = curState.rng.counter;
            const BattleContext preActionState = curState;

            edgeTaken.action.execute(curState);

            actionStack.push_back(edgeTaken.action);

            const bool rngChanged = curState.rng.counter != rngCounterBefore;
            if (rngChanged) {
                // Convert child to random node and seed first observed outcome as an edge
                edgeTaken.node.isRandomNode = true;
                edgeTaken.node.stochasticAction = edgeTaken.action;
                edgeTaken.node.outcomesGenerated = 0;

                // First outcome child corresponds to no extra RNG advance beyond what action consumed on creation path
                search::BattleSearcher::Edge observedOutcomeEdge;
                observedOutcomeEdge.rngAdvanceSteps = 0;
                edgeTaken.node.edges.push_back(std::move(observedOutcomeEdge));
                ++edgeTaken.node.outcomesGenerated;

                // Descend through random node into the observed outcome child
                searchStack.push_back(&edgeTaken.node);
                searchStack.push_back(&edgeTaken.node.edges.back().node);
            } else {
                // Deterministic: descend normally
                searchStack.push_back(&edgeTaken.node);
            }

            rolloutToEnd(curState, actionStack);
            updateFromPlayout(searchStack, actionStack, curState);
            return;

        } else {
            const auto selectIdx = selectBestEdgeToSearch(curNode);
            auto &edgeTaken = curNode.edges[selectIdx];

            if (edgeTaken.node.isRandomNode) {
                // Transition into random node; do not execute action here
                actionStack.push_back(edgeTaken.action);
                searchStack.push_back(&edgeTaken.node);
            } else {
                edgeTaken.action.execute(curState);
                actionStack.push_back(edgeTaken.action);
                searchStack.push_back(&edgeTaken.node);
            }
        }
    }
}

void search::BattleSearcher::updateFromPlayout(const std::vector<Node *> &stack, const std::vector<Action> &actionStack, const BattleContext &endState) {
    const auto evaluation = evaluateEndState(endState);

    if (evaluation > bestActionValue) {
        bestActionSequence = actionStack;
        bestActionValue = evaluation;
        outcomePlayerHp = endState.player.curHp;
    }

    if (evaluation < minActionValue) {
        minActionValue = evaluation;
    }

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
    
    if (edge.node.simulationCount == 0) {
        return std::numeric_limits<double>::infinity();
    }

    double qualityValue = edge.node.evaluationSum / edge.node.simulationCount;

    double explorationValue = explorationParameter *
            std::sqrt(std::log(parent.simulationCount+1) / (edge.node.simulationCount+1));

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

void search::BattleSearcher::expandRandomOutcome(search::BattleSearcher::Node &randomNode, BattleContext &curState) {
#ifdef sts_asserts
    assert(randomNode.isRandomNode);
#endif
    // Build state from current traversal state; at entry to a random node, curState is at pre-action
    BattleContext bc = curState;
    const int advance = randomNode.outcomesGenerated;
    for (int i = 0; i < advance; ++i) {
        bc.rng.randomBoolean();
    }
    randomNode.stochasticAction.execute(bc);

    search::BattleSearcher::Edge e;
    e.rngAdvanceSteps = advance;
    randomNode.edges.push_back(std::move(e));
    ++randomNode.outcomesGenerated;

    // Update the traversal state to match this realized outcome
    curState = bc;
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
            // Parent is random node: we are at pre-action state; apply RNG advances and execute stochastic action
            for (int i = 0; i < edgeRef.rngAdvanceSteps; ++i) {
                bc.rng.randomBoolean();
            }
            const auto &stochAction = parentNode->stochasticAction;
            stochAction.execute(bc);
        } else {
            edgeRef.action.execute(bc);
        }

        curStack.push_back( {&edgeRef.node, new BattleContext(bc), 0} );
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
            os << "(" << x.first.node.simulationCount << ")";
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
