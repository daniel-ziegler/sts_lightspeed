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
            bool isRandomNode = false;
            Action stochasticAction;
            int outcomesGenerated = 0;

            Node() = default;
            Node(const Node& other);
            Node(Node&&) = default;
            Node& operator=(const Node& other);
            Node& operator=(Node&&) = default;
        };

        struct Edge {
            Action action;
            Node node;
            int rngAdvanceSteps = 0;
        };

        std::unique_ptr<const BattleContext> rootState;
        Node root;

        EvalFnc evalFnc;
        double explorationParameter = 3*sqrt(2);

        std::default_random_engine randGen;

        std::vector<Node*> searchStack;
        std::vector<Action> actionStack;
        
        SimpleAgent rolloutAgent;

        explicit BattleSearcher(const BattleContext &bc, EvalFnc evalFnc=&evaluateEndState);

        // public methods
        void search(int64_t simulations);
        void step();
        Action getBestAction() const;
        const std::vector<Edge>& getRootEdges() const { return root.edges; }

        // private helpers
        void updateFromPlayout(const std::vector<Node*> &stack, const std::vector<Action> &actionStack, const BattleContext &endState);
        [[nodiscard]] bool isTerminalState(const BattleContext &bc) const;

        double evaluateEdge(const Node &parent, int edgeIdx);
        int selectBestEdgeToSearch(const Node &cur);
        int selectFirstActionForLeafNode(const Node &leafNode);

        void rolloutToEnd(BattleContext &state, std::vector<Action> &actionStack);

        void enumerateActionsForNode(Node &node, const BattleContext &bc);
        void enumerateCardActions(Node &node, const BattleContext &bc);
        void enumeratePotionActions(Node &node, const BattleContext &bc);
        void enumerateCardSelectActions(Node &node, const BattleContext &bc);
        void expandRandomOutcome(Node &randomNode, BattleContext &curState);
        static double evaluateEndState(const BattleContext &bc);

        void printSearchTree(std::ostream &os, int levels);
        void printSearchStack(std::ostream &os, bool skipLast=false);
    };

    extern thread_local BattleSearcher *g_debug_scum_search;

}


#endif //STS_LIGHTSPEED_BATTLESEARCHER_H
