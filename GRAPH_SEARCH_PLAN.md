# Graph Search Deduplication Implementation Plan

## Problem

We need graph search (sharing nodes when different action sequences reach identical states) but can't use `std::shared_ptr<Node>` in Edge because pybind11 creates Python wrappers that conflict with C++ lifetime management, causing memory corruption.

## Solution: State Hashing + Node Pool with Raw Pointers

### Architecture

Use **raw pointers** instead of shared_ptr, with BattleSearcher owning all nodes:

```cpp
struct Edge {
    Action action;
    Node* node;  // Raw pointer (safe for pybind11)
    int rngAdvanceSteps = 0;
};

// In BattleSearcher:
std::vector<std::unique_ptr<Node>> allNodes;  // Node ownership pool
std::unordered_map<size_t, Node*> stateToNode;  // State hash -> node mapping
```

**Why this works:**
- BattleSearcher owns all nodes via `allNodes`
- Raw pointers in Edge don't trigger pybind11 lifetime management
- Python can access nodes safely as long as BattleSearcher is alive
- Destructor cleans up all nodes deterministically

### State Hashing

Implement `size_t hashBattleState(const BattleContext& bc)` that hashes:

**Essential state (must match for deduplication):**
- Player: curHp, maxHp, block, energy, powers, relics effects
- Monsters: curHp, maxHp, block, powers, move intent
- Cards: hand contents, draw pile, discard pile, exhaust pile
- Turn number, cards played this turn
- Potions

**Excluded from hash:**
- RNG counter (will differ even for "same" state due to different paths)
- Action history
- Debug counters

**RNG Handling:**
- States reached via different RNG paths should still deduplicate
- The `randomnessBase` field handles RNG branching separately
- Hash represents "observable game state" not "complete simulator state"

### Node Creation with Deduplication

```cpp
Node* getOrCreateNode(const BattleContext& state) {
    size_t hash = hashBattleState(state);

    auto it = stateToNode.find(hash);
    if (it != stateToNode.end()) {
        // State already seen - reuse existing node (graph merge!)
        return it->second;
    }

    // New state - create new node
    allNodes.push_back(std::make_unique<Node>());
    Node* newNode = allNodes.back().get();
    stateToNode[hash] = newNode;
    return newNode;
}
```

### Integration into step()

**Current (tree search):**
```cpp
auto &edgeTaken = curNode.edges[selectIdx];
edgeTaken.action.execute(curState);
searchStack.push_back(&edgeTaken.node);  // node by value
```

**New (graph search):**
```cpp
auto &edgeTaken = curNode.edges[selectIdx];
edgeTaken.action.execute(curState);

if (edgeTaken.node == nullptr) {
    // First time visiting this edge - create or find node
    edgeTaken.node = getOrCreateNode(curState);
}

searchStack.push_back(edgeTaken.node);  // node by pointer
```

### Random Nodes

Random nodes already have special handling:
- They represent stochastic actions that branch into RNG outcomes
- Each outcome is a separate edge with `rngAdvanceSteps`
- Outcomes can deduplicate if they reach the same state (e.g., "shuffle deck" outcomes)

```cpp
void expandRandomOutcome(Node &randomNode, BattleContext &curState) {
    const std::uint64_t base = randomNode.randomnessBase;
    curState.rng = Random(base + randomNode.outcomesGenerated);
    randomNode.stochasticAction.execute(curState);

    // Create or find node for this outcome
    Node* outcomeNode = getOrCreateNode(curState);

    Edge outcomeEdge;
    outcomeEdge.action = Action{};
    outcomeEdge.node = outcomeNode;
    outcomeEdge.rngAdvanceSteps = randomNode.outcomesGenerated;

    randomNode.edges.push_back(outcomeEdge);
    ++randomNode.outcomesGenerated;
}
```

### Edge Initialization

Change Edge struct:
```cpp
struct Edge {
    Action action;
    Node* node = nullptr;  // Initialize to nullptr
    int rngAdvanceSteps = 0;
};
```

When enumerating actions:
```cpp
void enumerateCardActions(Node &node, const BattleContext &bc) {
    // ...
    node.edges.push_back({Action(ActionType::CARD, handIdx, tIdx)});
    // node pointer stays nullptr until first visit
}
```

### Cleanup

```cpp
~BattleSearcher() {
    stateToNode.clear();
    allNodes.clear();  // unique_ptrs automatically delete
}
```

## Implementation Steps

1. **Add state hashing:**
   - Implement `hashBattleState(const BattleContext&)`
   - Test hash collisions on known states

2. **Update data structures:**
   - Change `Edge::node` from `Node node` to `Node* node = nullptr`
   - Add `std::vector<std::unique_ptr<Node>> allNodes`
   - Add `std::unordered_map<size_t, Node*> stateToNode`

3. **Add `getOrCreateNode()`:**
   - Implement node creation with deduplication

4. **Update step() function:**
   - Lazy node creation on first edge visit
   - Use `getOrCreateNode()` for deduplication

5. **Update other functions:**
   - `expandRandomOutcome()`: use `getOrCreateNode()`
   - `getBestAction()`: handle nullptr nodes
   - `evaluateEdge()`: handle nullptr nodes
   - All SearchAgent.cpp references

6. **Update pybind11 bindings:**
   - Change from `.def_readonly("node", ...)` to use raw pointer semantics
   - Test that Python can access nodes safely

7. **Test and validate:**
   - Use Python test to verify deduplication
   - Check memory usage reduction
   - Ensure no crashes with CommunicationMod

## Expected Benefits

- **Memory reduction:** States reached via multiple paths share nodes
- **Better exploration:** Visit counts properly reflect all paths to a state
- **Correct statistics:** UCT values account for all visits to a state
- **No pybind11 issues:** Raw pointers don't trigger Python lifetime management

## Testing Strategy

Use Python to:
1. Create a simple battle with deterministic outcomes
2. Force multiple paths to same state (e.g., "Strike then Defend" vs "Defend then Strike")
3. Verify that nodes are shared (check memory addresses or visit counts)
4. Run full integration with CommunicationMod to ensure no crashes
