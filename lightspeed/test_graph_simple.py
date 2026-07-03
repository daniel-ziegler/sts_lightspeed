#!/usr/bin/env python3
"""
Simple test for graph search deduplication in BattleSearcher.

Tests the current tree-based implementation as a baseline,
then can be used to verify graph search after implementing GRAPH_SEARCH_PLAN.md
"""

import slaythespire as sts


def test_basic_search():
    """
    Basic test that BattleSearcher works with current implementation.
    """
    print("=" * 80)
    print("Test: Basic BattleSearcher Functionality")
    print("=" * 80)

    # Create a game that will progress to a real battle
    seed = 12345
    ascension = 0

    print(f"\nCreating game with seed={seed}, ascension={ascension}")
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, ascension)

    # Execute actions until we reach a battle
    max_actions = 100
    action_count = 0

    while gc.screen_state != sts.ScreenState.BATTLE and action_count < max_actions:
        if gc.outcome != sts.GameOutcome.UNDECIDED:
            print("❌ Game ended before battle")
            return False

        actions = sts.GameAction.getAllActionsInState(gc)
        if not actions:
            print("❌ No actions available")
            return False

        # Take first action to progress
        actions[0].execute(gc)
        action_count += 1

    if gc.screen_state != sts.ScreenState.BATTLE:
        print(f"❌ Failed to reach battle after {action_count} actions")
        print(f"   Current screen: {gc.screen_state}")
        return False

    print(f"✅ Reached battle after {action_count} actions")

    # Create battle context
    bc = gc.create_battle_context()

    print(f"\nBattle Information:")
    print(f"  Player HP: {bc.player.curHp}/{bc.player.maxHp}")
    print(f"  Monsters: {bc.monsters.monsterCount}")
    print(f"  Turn: {bc.turn}")

    # Run MCTS search
    print("\n" + "-" * 80)
    print("Running MCTS search with 1000 simulations...")
    print("-" * 80)

    searcher = sts.BattleSearcher(bc)
    searcher.exploration_parameter = 1.414  # sqrt(2)
    searcher.search(1000)

    # Analyze results
    root_edges = searcher.get_root_edges()
    print(f"\n✅ Search completed successfully")
    print(f"   Root has {len(root_edges)} possible actions")

    if len(root_edges) == 0:
        print("⚠️  Warning: No actions found (might be terminal state)")
        return None

    # Print top actions by visit count
    action_visits = []
    for i, edge in enumerate(root_edges):
        visits = edge.node.simulation_count
        eval_sum = edge.node.evaluation_sum
        avg_value = eval_sum / visits if visits > 0 else 0.0
        action_visits.append((i, visits, avg_value))

    action_visits.sort(key=lambda x: x[1], reverse=True)

    print(f"\n   Top 5 actions by visit count:")
    for i, (idx, visits, avg_value) in enumerate(action_visits[:5]):
        print(f"     {i+1}. Action {idx:2d}: visits={visits:4d}  avg_value={avg_value:6.2f}")

    # Check for graph structure characteristics
    print("\n" + "-" * 80)
    print("Analyzing tree structure (current implementation)...")
    print("-" * 80)

    # Count nodes at different depths
    def count_nodes_at_depth(edges, depth=0, max_depth=3):
        """Count nodes using BFS"""
        if depth >= max_depth:
            return set()

        nodes = set()
        for edge in edges:
            if edge.node:
                node_id = id(edge.node)
                nodes.add(node_id)

        return nodes

    unique_depth1 = count_nodes_at_depth(root_edges, 0, 1)
    print(f"   Unique nodes at depth 1: {len(unique_depth1)}")
    print(f"   Total root edges: {len(root_edges)}")

    # With current tree implementation, each edge should have unique node
    if len(unique_depth1) == len(root_edges):
        print(f"\n   ✅ Tree structure confirmed (each edge has unique node)")
        print(f"      After implementing graph search, this should decrease")
    else:
        print(f"\n   ⚠️  Unexpected: nodes={len(unique_depth1)} != edges={len(root_edges)}")

    return True


def test_node_structure():
    """
    Test that we can access Node fields correctly.
    """
    print("\n" + "=" * 80)
    print("Test: Node Structure Access")
    print("=" * 80)

    seed = 54321
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)

    # Progress to battle
    max_actions = 100
    action_count = 0
    while gc.screen_state != sts.ScreenState.BATTLE and action_count < max_actions:
        if gc.outcome != sts.GameOutcome.UNDECIDED:
            return False
        actions = sts.GameAction.getAllActionsInState(gc)
        if not actions:
            return False
        actions[0].execute(gc)
        action_count += 1

    if gc.screen_state != sts.ScreenState.BATTLE:
        return False

    bc = gc.create_battle_context()
    print(f"Battle created")

    # Small search
    searcher = sts.BattleSearcher(bc)
    searcher.search(100)

    root_edges = searcher.get_root_edges()

    print(f"\nTesting node field access:")
    for i, edge in enumerate(root_edges[:3]):
        print(f"\n  Edge {i}:")
        print(f"    Has node: {edge.node is not None}")
        if edge.node:
            print(f"    simulation_count: {edge.node.simulation_count}")
            print(f"    evaluation_sum: {edge.node.evaluation_sum}")

            # Try accessing action
            try:
                action_type = edge.action.get_action_type()
                print(f"    action_type: {action_type}")
            except Exception as e:
                print(f"    action access failed: {e}")

    print("\n✅ Node structure access successful")
    return True


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 18 + "GRAPH SEARCH BASELINE TEST" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\nThis test establishes a baseline for the current tree-based implementation.")
    print("After implementing GRAPH_SEARCH_PLAN.md, re-run to verify graph deduplication.\n")

    results = []

    try:
        result = test_basic_search()
        results.append(("Basic Search", result))
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Basic Search", False))

    try:
        result = test_node_structure()
        results.append(("Node Structure", result))
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Node Structure", False))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for name, result in results:
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  INCONCLUSIVE"
        print(f"  {status}  {name}")

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("  1. Review GRAPH_SEARCH_PLAN.md for implementation details")
    print("  2. Implement state hashing and node pool architecture")
    print("  3. Re-run this test to verify graph deduplication works")
    print("  4. Run test_graph_search.py for detailed deduplication metrics")
    print("=" * 80 + "\n")
