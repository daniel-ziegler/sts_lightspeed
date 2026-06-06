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

    // Deterministic search-graph telemetry: counts are exact properties of the search
    // (machine/thread independent), summable across searchers.
    struct SearchStats {
        std::int64_t steps = 0;                  // simulations run
        std::int64_t nodesCreated = 0;           // new nodes allocated via dedup misses
        std::int64_t detTranspositions = 0;      // deterministic-edge expansion hit an existing node
        std::int64_t chanceOutcomesSampled = 0;  // chance-node widening samples executed
        std::int64_t chanceSiblingReuse = 0;     // sampled outcome collided with an existing sibling
        std::int64_t chanceTranspositions = 0;   // sampled outcome dedup-hit a non-sibling node
        std::int64_t depthSum = 0;               // in-tree path length at each simulation's end
        std::int64_t chanceDepthSum = 0;         // chance nodes on the path at each simulation's end
        void add(const SearchStats &o) {
            steps += o.steps; nodesCreated += o.nodesCreated; detTranspositions += o.detTranspositions;
            chanceOutcomesSampled += o.chanceOutcomesSampled; chanceSiblingReuse += o.chanceSiblingReuse;
            chanceTranspositions += o.chanceTranspositions;
            depthSum += o.depthSum; chanceDepthSum += o.chanceDepthSum;
        }
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

        std::unique_ptr<BattleContext> rootState;
        Node* root = nullptr;   // points into allNodes; set by resetForSearch (fresh) or rerootAt (reuse)

        // Graph search: node pool and state deduplication.
        // stateToNode buckets by search-hash; collisions are resolved by equalForSearch,
        // so distinct states that happen to share a hash are never merged.
        // The pool persists across a battle's decisions: each search reuses allNodes[0..poolUsed)
        // and only grows the vector when it needs more, so nodes aren't freed and reallocated per move.
        std::vector<std::unique_ptr<Node>> allNodes;  // Pool of all created nodes
        std::size_t poolUsed = 0;                     // nodes claimed by the current search

        // Open-addressed dedup table. Backed by a single vector and cleared with one memset per
        // search; replaces unordered_map<size_t, vector<Node*>> -- no per-bucket vector, no per-
        // entry heap node. Sized once at construction to comfortably hold any reasonable search;
        // hash collisions are resolved by linear probing + equalForSearch (the dedup invariant).
        struct DedupSlot { std::size_t hash; Node* node; };  // node == nullptr means empty
        std::vector<DedupSlot> stateToNode;
        std::size_t stateToNodeMask = 0;              // stateToNode.size() - 1 (power-of-two size)

        EvalFnc evalFnc;
        EvalWeights evalWeights;
        double explorationParameter = 9.9;   // tuned default

        // Double Progressive Widening for chance nodes: after n visits a chance node may
        // hold at most ceil(chanceWideningC * (n+1)^chanceWideningAlpha) distinct outcomes.
        double chanceWideningC = 4.6;        // tuned default
        double chanceWideningAlpha = 0.37;   // tuned default

        std::default_random_engine randGen;

        std::vector<Node*> searchStack;        // descent path; doubles as the cycle-guard set (linear scan, depth is small)
        std::vector<Action> actionStack;
        
        SimpleAgent rolloutAgent;
        BattleContext rolloutScratch;   // reused playout buffer: copy-assigning into it keeps the card-pile vector capacity, avoiding a fresh allocation per rollout
        // Same idiom for the two expansion paths: candidate states are built in persistent
        // scratch buffers, so duplicate outcomes (dedup hits, >half of chance samples) cost no
        // allocation. getOrCreateNode steals the buffers only when a node is actually created.
        BattleContext expandScratch;
        BattleContext widenScratch;

        explicit BattleSearcher(const BattleContext &bc, EvalFnc evalFnc=nullptr);
        ~BattleSearcher();

        // public methods
        void setRoot(const BattleContext &bc);      // fresh reset: clear pool/dedup, alloc new root, reseed rng
        void rerootAt(Node* newRoot);               // tree reuse: keep pool/dedup; root = newRoot; reseed rng
        void search(int64_t simulations);
        void searchForMicros(int64_t maxMicros);   // run steps until the wall-clock budget (microseconds) is spent
        void step();
        Action getBestAction() const;
        const std::vector<Edge>& getRootEdges() const { return root->edges; }

        // private helpers
        bool resetForSearch();   // reset node pool/root for a fresh search; returns false if the root is already terminal
        void updateFromPlayout(const std::vector<Node*> &stack, const std::vector<Action> &actionStack, const BattleContext &endState);
        [[nodiscard]] bool isTerminalState(const BattleContext &bc) const;

        Node* allocNode();   // claim a node from the pool (recycling a reset one, or growing the pool)

        // Graph search deduplication
        Node* getOrCreateNode(BattleContext &&state);
        bool lastNodeWasCreated = false;   // whether the last getOrCreateNode allocated a new node

        SearchStats stats;

        double evaluateEdge(const Node &parent, int edgeIdx, double logParentVisits);
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
