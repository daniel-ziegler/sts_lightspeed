//
// Created by keega on 9/17/2021.
//

#ifndef STS_LIGHTSPEED_BATTLESEARCHER_H
#define STS_LIGHTSPEED_BATTLESEARCHER_H

#include "sim/search/Action.h"

#include <functional>
#include <memory>
#include <random>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "sim/search/SimpleAgent.h"

namespace sts::search {

    typedef std::function<double (const BattleContext&)> EvalFnc;

    // Tunable weights for evaluateEndState (the value backed up by the search).
    // Defaults are the tuned config (Optuna, validated best on 2000 states @5000).
    struct EvalWeights {
        double winBonus = 53.0;
        double potionWeight = 11.0;
        double victoryTurnPenalty = 0.01;
        double monsterDamageWeight = 37.0;
        double aliveWeight = 3.4;
        double energyWasteWeight = 1.75;
        double drawWeight = 0.03;
        double turnSurvivalWeight = 1.5;
    };

    // to find a solution to a battle with tree pruning
    struct BattleSearcher {
        class Edge;
        struct Node {
            std::int64_t simulationCount = 0;
            double evaluationSum = 0;
            std::vector<Edge> edges;
            BattleContext state;                // game state at this node (decision nodes; chance nodes read parent->state)
            bool isRandomNode = false;
            Action stochasticAction;            // chance node: the action that consumed RNG
            Node* parent = nullptr;             // chance node: spawning decision node (source of pre-action state)
            int outcomesGenerated = 0;          // chance node: number of RNG outcomes sampled so far
            std::uint64_t randomnessBase = 0;   // chance node: RNG base, sampled from parent's pre-action RNG
        };

        struct Edge {
            Action action;
            Node* node = nullptr;               // child (shared across paths via transposition)
            int rngAdvanceSteps = 0;            // chance outcome edge: the N used for Random(base + N)
            std::int32_t visitCount = 0;        // times this edge was traversed (UCB / DPW reselection)
        };

        std::unique_ptr<const BattleContext> rootState;
        Node root;

        // Graph search: node pool and state deduplication.
        // stateToNode buckets by search-hash; collisions are resolved by equalForSearch,
        // so distinct states that happen to share a hash are never merged.
        std::vector<std::unique_ptr<Node>> allNodes;  // Pool of all created nodes
        std::unordered_map<size_t, std::vector<Node*>> stateToNode;

        EvalFnc evalFnc;
        EvalWeights evalWeights;
        double explorationParameter = 9.9;   // tuned default

        // Double Progressive Widening for chance nodes: after n visits a chance node may
        // hold at most ceil(chanceWideningC * (n+1)^chanceWideningAlpha) distinct outcomes.
        double chanceWideningC = 4.6;        // tuned default
        double chanceWideningAlpha = 0.37;   // tuned default

        std::default_random_engine randGen;

        std::vector<Node*> searchStack;
        std::vector<Action> actionStack;
        std::unordered_set<Node*> onPathSet;   // nodes on the current descent path (cycle guard)
        
        SimpleAgent rolloutAgent;

        explicit BattleSearcher(const BattleContext &bc, EvalFnc evalFnc=nullptr);
        ~BattleSearcher();

        // public methods
        void search(int64_t simulations);
        void step();
        Action getBestAction() const;
        const std::vector<Edge>& getRootEdges() const { return root.edges; }

        // private helpers
        void updateFromPlayout(const std::vector<Node*> &stack, const std::vector<Action> &actionStack, const BattleContext &endState);
        [[nodiscard]] bool isTerminalState(const BattleContext &bc) const;

        // Graph search deduplication
        Node* getOrCreateNode(const BattleContext &state);

        double evaluateEdge(const Node &parent, int edgeIdx);
        int selectBestEdgeToSearch(const Node &cur);

        void rolloutToEnd(BattleContext &state, std::vector<Action> &actionStack);

        void enumerateActionsForNode(Node &node, const BattleContext &bc);
        void enumerateCardActions(Node &node, const BattleContext &bc);
        void enumeratePotionActions(Node &node, const BattleContext &bc);
        void enumerateCardSelectActions(Node &node, const BattleContext &bc);
        Edge* selectChanceOutcome(Node &chance);
        double evaluateEndState(const BattleContext &bc) const;

        void printSearchTree(std::ostream &os, int levels);
        void printSearchStack(std::ostream &os, bool skipLast=false);
    };

    extern thread_local BattleSearcher *g_debug_scum_search;

}


#endif //STS_LIGHTSPEED_BATTLESEARCHER_H
