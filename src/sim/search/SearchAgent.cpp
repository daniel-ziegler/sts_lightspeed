//
// Created by keega on 9/19/2021.
//

#include "sim/search/SearchAgent.h"

#include <algorithm>
#include <atomic>
namespace sts::search { std::atomic<long> g_rerootExact{0}, g_rerootPermuted{0}, g_rerootMiss{0}; }

#include <sim/search/ExpertKnowledge.h>
#include <game/Game.h>
#include "sim/PrintHelpers.h"
#include "sim/search/BattleSearcher.h"

using namespace sts;

void search::SearchAgent::takeAction(GameContext &gc, GameAction a) {
    if (recordActions) {
        gameActionHistory.emplace_back(a.bits);
    }
    if (printActions) {
        std::cout << std::hex << a.bits << std::endl;
    }
    a.execute(gc);
}

void search::SearchAgent::takeAction(BattleContext &bc, search::Action a) {
    if (recordActions) {
        gameActionHistory.emplace_back(a.bits);
    }
    if (printActions) {
        std::cout << std::hex << a.bits << std::endl;
    }
    a.execute(bc);
}

void search::SearchAgent::playout(GameContext &gc) {
    paused = false;
    BattleContext bc;
    const auto seedStr = std::string(SeedHelper::getString(gc.seed));

    while (gc.outcome == GameOutcome::UNDECIDED && !paused) {
        if (gc.screenState == ScreenState::BATTLE) {
            if (recordActions) {
                battleStartIndices.push_back(static_cast<int>(gameActionHistory.size()));
            }

            bc = BattleContext();
            bc.init(gc);
            const auto battleEncounter = bc.encounter;

            playoutBattle(bc);
            bc.exitBattle(gc);

            if (logBattleOutcomes) {
                battleLog.push_back({gc.floorNum, gc.act, gc.curHp, gc.maxHp, gc.potionCount,
                                     static_cast<int>(gc.deck.size()), static_cast<int>(battleEncounter)});
            }
            continue;
        }

        if (verbosityLevel >= 2) {
            std::cout << gc << std::endl;
        }
        stepOutOfCombatPolicy(gc);
    }
}

static void printHelper(const BattleContext &bc, const search::Action &a) {
    a.printDesc(std::cout, bc) << " ";
    std::cout
            << " turn: " << bc.turn
            << " energy: " << bc.player.energy
            << " cardsPlayedThisTurn: " << bc.player.cardsPlayedThisTurn
            << " state: " << (bc.inputState == InputState::PLAYER_NORMAL ? "normal" : " probably card select")
            << std::endl;
    std::cout << bc << std::endl;
}

void search::SearchAgent::playoutBattle(BattleContext &bc) {
    // One searcher for the whole battle. Each decision tries to reroot at the chosen child's
    // subtree so its cached visit counts carry over as a head start; falls back to a full setRoot
    // when no matching subtree exists (chance outcome wasn't sampled) or the node pool got large.
    search::BattleSearcher searcher(bc);   // ctor's resetForSearch sets up root for the first iter
    const bool boss = isBossEncounter(bc.encounter);
    searcher.explorationParameter = explorationParameter;
    searcher.explorationParameterChance = explorationParameterChance;
    searcher.chanceWideningC = boss ? bossChanceWideningC : chanceWideningC;
    searcher.chanceWideningAlpha = boss ? bossChanceWideningAlpha : chanceWideningAlpha;
    searcher.endTurnWideningC = endTurnWideningC;
    searcher.endTurnWideningAlpha = endTurnWideningAlpha;
    searcher.evalWeights = evalWeights;

    // Cap reuse-driven pool growth so the dedup table's load factor stays low. Beyond this we do
    // a full setRoot, which recycles all nodes into the pool and drops the dedup map.
    constexpr std::size_t reusePoolCap = 32 * 1024;

    const sts::search::BattleSearcher::Edge *prevBestEdge = nullptr;

    while (bc.outcome == Outcome::UNDECIDED) {
        if (prevBestEdge != nullptr) {
            // Locate the new root in the existing tree. Rerooting adopts the node's stored
            // state, and the next decision's action indices execute on the REAL bc — so the
            // match must be exact including hand order (equalForSearch treats the hand as a
            // multiset, which is sound inside the tree but not at the tree/reality boundary).
            using Node = sts::search::BattleSearcher::Node;
            auto exactMatch = [&bc](const Node *n) {
                return n->state.equalForSearch(bc) &&
                       std::equal(n->state.cards.hand.begin(),
                                  n->state.cards.hand.begin() + n->state.cards.cardsInHand,
                                  bc.cards.hand.begin());
            };
            Node* candidate = nullptr;
            Node* permutedCandidate = nullptr;
            Node* edgeChild = prevBestEdge->node;
            if (edgeChild != nullptr && !edgeChild->isRandomNode) {
                if (exactMatch(edgeChild)) {
                    candidate = edgeChild;
                } else if (edgeChild->state.equalForSearch(bc)) {
                    permutedCandidate = edgeChild;
                }
            } else if (edgeChild != nullptr) {
                // Chance node: find the outcome whose realized state matches bc.
                for (auto &oe : edgeChild->edges) {
                    if (oe.node == nullptr) continue;
                    if (exactMatch(oe.node)) {
                        candidate = oe.node;
                        break;
                    }
                    if (permutedCandidate == nullptr && oe.node->state.equalForSearch(bc)) {
                        permutedCandidate = oe.node;
                    }
                }
            }
            g_rerootExact += (candidate != nullptr);
            g_rerootPermuted += (candidate == nullptr && permutedCandidate != nullptr);
            g_rerootMiss += (candidate == nullptr && permutedCandidate == nullptr);
            // permutedCandidate is counted but NOT used: rerooting into a node matching only up
            // to hand order (with uid-remapped action emission) gated NEGATIVE at deployment
            // (60.6% vs 62.7% control, 1000 paired seeds) -- reusing those stale subtrees loses
            // more than the fresh-search budget it saves.
            if (candidate != nullptr && searcher.allNodes.size() < reusePoolCap) {
                searcher.rerootAt(candidate);
            } else {
                searcher.setRoot(bc);
            }
            prevBestEdge = nullptr;   // bestEdge memory may be invalidated by setRoot above
        }

        const double bossMultiplier = isBossEncounter(bc.encounter) ? bossSimulationMultiplier : 1.0;
        if (searchTimeMicros > 0) {
            searcher.searchForMicros(static_cast<std::int64_t>(bossMultiplier * searchTimeMicros));
        } else {
            searcher.search(static_cast<std::int64_t>(bossMultiplier * simulationCountBase));
        }
        simulationCountTotal += searcher.root->simulationCount;

        const auto &rootNode = *searcher.root;
        if (rootNode.edges.empty()) {
            break;
        }

        const sts::search::BattleSearcher::Edge *bestEdge = nullptr;
        std::int32_t maxVisits = -1;
        for (const auto &edge : rootNode.edges) {
            if (edge.visitCount > maxVisits) {
                maxVisits = edge.visitCount;
                bestEdge = &edge;
            }
        }

        assert(bestEdge != nullptr);

        if (verbosityLevel == 2) {
            printHelper(bc, bestEdge->action);
        } else if (verbosityLevel == 1) {
            printConciseAction(bc, bestEdge->action);
        }

        takeAction(bc, bestEdge->action);
        prevBestEdge = bestEdge;   // valid for next iter's reroot lookup (no setRoot between here and there)
    }
    searchStats.add(searcher.stats);
}

void search::SearchAgent::stepThroughSolution(BattleContext &bc, std::vector<search::Action> &actions) {
    for (int i = 0; i < stepsWithSolution; ++i) {
        if (actions.empty()) {
            break;
        }

        auto &a = actions.back();
        if (verbosityLevel == 2) {
            printHelper(bc, a);
        } else if (verbosityLevel == 1) {
            printConciseAction(bc, a);
        }

        takeAction(bc, a);
        actions.pop_back();
    }
}

void search::SearchAgent::stepThroughSearchTree(BattleContext &bc, const search::BattleSearcher &s) {
    const search::BattleSearcher::Node *curNode = s.root;
    for (int actionCount = 0; actionCount < stepsNoSolution; ++actionCount) {
        if (bc.outcome != Outcome::UNDECIDED) {
            break;
        }

        std::int32_t maxSimulations = -1;
        const sts::search::BattleSearcher::Edge *maxEdge = nullptr;

        for (const auto &edge : curNode->edges) {
            if (edge.visitCount > maxSimulations) {
                maxSimulations = edge.visitCount;
                maxEdge = &edge;
            }
        }

        if (maxEdge == nullptr) {
            break;
        }

        if (verbosityLevel == 2) {
            printHelper(bc, maxEdge->action);
        } else if (verbosityLevel == 1) {
            printConciseAction(bc, maxEdge->action);
        }

        takeAction(bc, maxEdge->action);
        curNode = maxEdge->node;
    }
}

GameAction search::SearchAgent::pickRandomAction(const GameContext &gc) {
    std::vector<GameAction> possibleActions(GameAction::getAllActionsInState(gc));
    std::uniform_int_distribution<int> distr(0, static_cast<int>(possibleActions.size())-1);
    const int randomChoice = distr(rng);
    return possibleActions[randomChoice];
}

void search::SearchAgent::stepOutOfCombatPolicy(GameContext &gc) {
    ++stepCount;
    GameAction action = pickOutOfCombatAction(gc);
    takeAction(gc, action);
}

GameAction search::SearchAgent::pickOutOfCombatAction(const GameContext &gc) {
    switch (gc.screenState) {
        case ScreenState::EVENT_SCREEN:
            return pickEventAction(gc);

        case ScreenState::REWARDS:
            return pickRewardsAction(gc);

        case ScreenState::TREASURE_ROOM: {
            // idx1 0 = open chest, 1 = skip
            bool takeChest = true;
            if (gc.relics.has(RelicId::CURSED_KEY)) {
                takeChest = gc.info.chestSize == ChestSize::LARGE;
            }
            return GameAction(takeChest ? 0 : 1);
        }

        case ScreenState::SHOP_ROOM: {
            for (int i = 0; i < 3; ++i) {
                if (gc.info.shop.relicPrice(i) != -1 &&  gc.gold >= gc.info.shop.relicPrice(i)) {
                    return GameAction(GameAction::RewardsActionType::RELIC, i);
                }
            }
            return pickRandomAction(gc);
        }

        case ScreenState::CARD_SELECT:
            return pickCardSelectAction(gc);

        case ScreenState::BOSS_RELIC_REWARDS: {
            int best = 10000;
            int bestIdx = 0;

            for (int i = 0; i < 3; ++i) {
                int value = search::Expert::getBossRelicOrdering(gc.info.bossRelics[i]);
                if (value < best) {
                    best = value;
                    bestIdx = i;
                }
            }
            return GameAction(GameAction::RewardsActionType::RELIC, bestIdx);
        }

        case ScreenState::REST_ROOM: {
            if (gc.curHp > 50 && gc.deck.getUpgradeableCount() > 0 && !gc.hasRelic(RelicId::FUSION_HAMMER)) {
                return GameAction(1);
            } else if (gc.curHp < 15 && !gc.relics.has(RelicId::COFFEE_DRIPPER)){
                return GameAction(0);
            } else {
                return pickRandomAction(gc);
            }
        }

        case ScreenState::BATTLE:
        case ScreenState::INVALID:
            assert(false);
            return GameAction();

        case ScreenState::MAP_SCREEN:
        default:
            return pickRandomAction(gc);
    }
}

GameAction search::SearchAgent::pickCardSelectAction(const GameContext &gc) {
    fixed_list<std::pair<int,int>, Deck::MAX_SIZE> selectOrder;

    for (int i = 0; i < gc.info.toSelectCards.size(); ++i) {
        const auto &c = gc.info.toSelectCards[i].card;

        auto playOrder = search::Expert::getPlayOrdering(c.getId());
        auto obtainWeight = search::Expert::getObtainWeight(c.getId());

        switch (gc.info.selectScreenType) {
            case CardSelectScreenType::TRANSFORM:
            case CardSelectScreenType::TRANSFORM_UPGRADE:
                if (c.getType() == CardType::CURSE) {
                    selectOrder.push_back( {i, playOrder} );
                } else {
                    selectOrder.push_back( {i, obtainWeight} );
                }
                break;

            case CardSelectScreenType::BONFIRE_SPIRITS:
            case CardSelectScreenType::REMOVE:
                selectOrder.push_back( {i, -obtainWeight} );
                break;

            case CardSelectScreenType::UPGRADE:
            case CardSelectScreenType::DUPLICATE:
            case CardSelectScreenType::OBTAIN:
            case CardSelectScreenType::BOTTLE:
                selectOrder.push_back( {i, -obtainWeight} );
                break;

            case CardSelectScreenType::INVALID:
            default:
                selectOrder.push_back({i, 0});
                break;
        }
    }
    std::sort(selectOrder.begin(), selectOrder.end(), [](auto a, auto b) { return a.second < b.second; });
    return GameAction(selectOrder.front().first);
}

GameAction search::SearchAgent::pickRewardsAction(const GameContext &gc) {
    auto &r = gc.info.rewardsContainer;
    if (r.goldRewardCount > 0) {
        return GameAction(GameAction::RewardsActionType::GOLD, 0);

    } else if (r.relicCount > 0) {
        return GameAction(GameAction::RewardsActionType::RELIC, 0);

    } else if (r.potionCount > 0) {
        return GameAction(GameAction::RewardsActionType::POTION, 0);

    } else if (r.cardRewardCount == 0) {
        return GameAction(GameAction::RewardsActionType::SKIP);

    } else {
        if (pauseOnCardReward) {
            paused = true;
            return GameAction();
        }
        return pickWeightedCardRewardAction(gc);
    }
}

double getAvgDeckWeight(const GameContext &gc) {
    int sum = 0;
    for (const auto &c : gc.deck.cards) {
        sum += search::Expert::getObtainWeight(c.getId(), c.isUpgraded());
    }
    return (double) sum / gc.deck.size();
}

GameAction search::SearchAgent::pickWeightedCardRewardAction(const GameContext &gc) {
    auto &r = gc.info.rewardsContainer;
    for (int rIdx = r.cardRewardCount-1; rIdx >= 0; --rIdx) {

        const auto deckWeight = getAvgDeckWeight(gc);
        if (verbosityLevel >= 2) {
            std::cout << "evaluating card reward " << rIdx << " avgDeckWeight: " << deckWeight << std::endl;
        }
        fixed_list<std::pair<int,double>,4> weights;
        double weightSum = 0;
        for (int cIdx = 0; cIdx < r.cardRewards[rIdx].size(); ++cIdx) {
            constexpr double act1AttackMultiplier = 1.4;

            const auto &c = r.cardRewards[rIdx][cIdx];
            double weight = std::pow(search::Expert::getObtainWeight(c.getId(), c.isUpgraded()), 1.2);
            if (gc.act == 1 && c.getType() == CardType::ATTACK) {
                weight *= act1AttackMultiplier;
            }

            weights.push_back({cIdx, weight});
            weightSum += weight;

            if (verbosityLevel >= 2) {
                std::cout << "card:" << r.cardRewards[rIdx][cIdx] << " eval: " << weight << std::endl;
            }
        }

        // choose a weighted card
        int selection = 0;
        {
            std::uniform_real_distribution<double> distr(0,weightSum);
            double roll = distr(rng);
            double acc = 0;
            for (int i = 0; i < weights.size(); ++i) {
                acc += weights[i].second;
                if (roll <= acc) {
                    selection = weights[i].first;
                }
            }
        }

        bool skipCard = true;
        {
            std::uniform_real_distribution<double> distr(0,weights[selection].second + deckWeight*0.6);
            double roll = distr(rng);
            if (roll < weights[selection].second) {
                skipCard = false;
            }
        }

        if (skipCard) {
            if (gc.hasRelic(RelicId::SINGING_BOWL)) {
                return GameAction(GameAction::RewardsActionType::CARD, rIdx, 5);
            } else {
                return GameAction(GameAction::RewardsActionType::SKIP);
            }

        } else {
            return GameAction(GameAction::RewardsActionType::CARD, rIdx, weights[selection].first);
        }
    }
    return GameAction();
}

GameAction search::SearchAgent::pickEventAction(const GameContext &gc) {
    switch (gc.curEvent) {

        case Event::NEOW:
            if (gc.info.neowRewards[1].d == Neow::Drawback::CURSE || gc.info.neowRewards[2].d == Neow::Drawback::CURSE) {
                return GameAction(0);
            } else {
                return pickRandomAction(gc);
            }

        case Event::NOTE_FOR_YOURSELF:
        case Event::THE_DIVINE_FOUNTAIN:
            return GameAction(0);

        case Event::BIG_FISH:
            return GameAction(1);

        case Event::GOLDEN_IDOL: {
            if (gc.hasRelic(RelicId::GOLDEN_IDOL)) {
                return GameAction(4);
            } else {
                return GameAction(0);
            }
        }

        case Event::GHOSTS:
        case Event::MASKED_BANDITS:
            return GameAction(0);

        case Event::CURSED_TOME:
            if (gc.info.eventData == 0) {
                return GameAction(0);
            } else {
                return GameAction(gc.info.eventData+1);
            }

        case Event::KNOWING_SKULL:
            return GameAction(3);

        default:
            return pickRandomAction(gc);
    }
}

void search::SearchAgent::printConciseAction(const BattleContext &bc, const Action &action) {
    
    // Print player status: HP and block
    std::cout << "Player: " << bc.player.curHp << "/" << bc.player.maxHp << "hp";
    if (bc.player.block > 0) {
        std::cout << "(" << bc.player.block << "blk)";
    }
    
    // Print monster status: HP and block for each alive monster
    std::cout << " | Monsters:";
    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        const auto &monster = bc.monsters.arr[i];
        if (monster.isAlive()) {
            std::cout << " " << monster.curHp << "/" << monster.maxHp << "hp";
            if (monster.block > 0) {
                std::cout << "(" << monster.block << "blk)";
            }
        }
    }
    std::cout << " ";

    // Print action description
    action.printDesc(std::cout, bc);

    std::cout << std::endl;
}
