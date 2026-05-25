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

#include "sim/search/SimpleAgent.h"

namespace sts::search {

    typedef std::function<double (const BattleContext&)> EvalFnc;

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
        double explorationParameter = 3*sqrt(2);

        // Double Progressive Widening for chance nodes: after n visits a chance node may
        // hold at most ceil(chanceWideningC * (n+1)^chanceWideningAlpha) distinct outcomes.
        double chanceWideningC = 1.0;
        double chanceWideningAlpha = 0.5;

        std::default_random_engine randGen;

        std::vector<Node*> searchStack;
        std::vector<Action> actionStack;
        
        SimpleAgent rolloutAgent;

        explicit BattleSearcher(const BattleContext &bc, EvalFnc evalFnc=&evaluateEndState);
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
        static double evaluateEndState(const BattleContext &bc);

        void printSearchTree(std::ostream &os, int levels);
        void printSearchStack(std::ostream &os, bool skipLast=false);
    };

    extern thread_local BattleSearcher *g_debug_scum_search;

}


#endif //STS_LIGHTSPEED_BATTLESEARCHER_H
