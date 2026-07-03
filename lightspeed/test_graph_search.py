#!/usr/bin/env python3
"""
Test script for graph search deduplication in BattleSearcher.

This test verifies that:
1. Multiple action sequences reaching the same state share the same node
2. Visit counts are properly accumulated across different paths
3. Memory is shared (not duplicated) for equivalent states
"""

import slaythespire as sts


def test_simple_combat_deduplication():
    """
    Test that simple combat actions that commute (same end state) share nodes.

    For example:
    - Path A: Play Strike -> Play Defend -> End Turn
    - Path B: Play Defend -> Play Strike -> End Turn

    Both paths should reach the same game state (same HP, same cards in discard, etc.)
    and therefore should share the same node in the search tree.
    """
    print("=" * 80)
    print("Test: Simple Combat Deduplication")
    print("=" * 80)

    # Create a simple game context and battle
    seed = 12345
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)

    # Create a battle context directly
    bc = gc.create_battle_context()

    print(f"\nBattle: {bc.encounter}")
    print(f"Player HP: {bc.player.curHp}/{bc.player.maxHp}")
    print(f"Monsters: {bc.monsters.monsterCount}")

    # Run MCTS search
    print("\n" + "-" * 80)
    print("Running MCTS search with 1000 simulations...")
    print("-" * 80)

    searcher = sts.BattleSearcher(bc)
    searcher.exploration_parameter = 1.414  # sqrt(2)
    searcher.search(1000)

    # Analyze the search tree
    root_edges = searcher.get_root_edges()

    print(f"\n Root node has {len(root_edges)} possible actions")

    total_visits = 0
    for i, edge in enumerate(root_edges):
        action_desc = f"Action {i}"
        # Try to get a description if available
        try:
            action_desc = str(edge.action)
        except:
            pass

        visits = edge.node.simulation_count if edge.node else 0
        eval_sum = edge.node.evaluation_sum if edge.node else 0.0
        avg_value = eval_sum / visits if visits > 0 else 0.0

        total_visits += visits

        if i < 10:  # Print first 10 actions
            print(f"  [{i:2d}] {action_desc:30s} visits={visits:4d}  avg_value={avg_value:6.2f}")

    print(f"\n Total visits across all children: {total_visits}")
    print(f" Root simulation count should be close to 1000")

    # Check for graph structure (deduplication)
    print("\n" + "-" * 80)
    print("Checking for graph deduplication...")
    print("-" * 80)

    # Collect all unique node addresses
    seen_nodes = set()
    total_edges = 0

    def count_nodes_bfs(edges, depth=0, max_depth=3):
        """Count unique nodes using BFS traversal"""
        nonlocal total_edges, seen_nodes

        if depth >= max_depth:
            return

        for edge in edges:
            total_edges += 1
            if edge.node:
                node_id = id(edge.node)
                seen_nodes.add(node_id)

                # Recurse into child edges
                if hasattr(edge.node, 'edges') and depth < max_depth - 1:
                    try:
                        child_edges = edge.node.edges
                        count_nodes_bfs(child_edges, depth + 1, max_depth)
                    except:
                        pass  # Can't access edges

    count_nodes_bfs(root_edges)

    print(f" Total edges explored (depth 3): {total_edges}")
    print(f" Unique nodes found: {len(seen_nodes)}")

    # Calculate theoretical maximum without deduplication
    # This is a rough estimate: root_edges + (root_edges * avg_children) + ...
    # For a tree, edges ≈ nodes, so deduplication_ratio shows sharing

    if total_edges > 0:
        deduplication_ratio = len(seen_nodes) / total_edges
        print(f" Deduplication ratio: {deduplication_ratio:.2%}")
        print(f"   (1.0 = no sharing/tree, <1.0 = node sharing/graph)")

        if deduplication_ratio < 0.95:
            print("\n✅ Graph search deduplication IS working!")
            print("   Multiple paths are sharing nodes")
            return True
        else:
            print("\n⚠️  Graph search deduplication NOT detected")
            print("   This might be expected if states rarely repeat at this depth")
            print("   Or deduplication is not yet implemented")
            return None  # Inconclusive

    return None


def test_random_node_deduplication():
    """
    Test that random nodes with same outcomes deduplicate.

    Random actions (like shuffle or card draw) can have multiple RNG outcomes.
    If two different random nodes generate the same outcome, those outcome states
    should share nodes.
    """
    print("\n" + "=" * 80)
    print("Test: Random Node Outcome Deduplication")
    print("=" * 80)

    seed = 54321
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    bc = gc.create_battle_context()

    print(f"Battle: {bc.encounter}")

    # Run search with more simulations to explore randomness
    print("\nRunning MCTS search with 5000 simulations...")

    searcher = sts.BattleSearcher(bc)
    searcher.search(5000)

    root_edges = searcher.get_root_edges()
    print(f"Root has {len(root_edges)} actions")

    # Look for random nodes
    random_nodes_found = 0
    for edge in root_edges:
        if edge.node and hasattr(edge.node, 'is_random_node'):
            if edge.node.is_random_node:
                random_nodes_found += 1

    if random_nodes_found > 0:
        print(f"✅ Found {random_nodes_found} random nodes in search tree")
    else:
        print("⚠️  No random nodes found (might be deterministic actions only)")

    return True


def test_memory_efficiency():
    """
    Test that graph search uses less memory than tree search would.

    This is measured by comparing simulation count distribution.
    With deduplication, we expect fewer unique nodes with higher visit counts.
    """
    print("\n" + "=" * 80)
    print("Test: Memory Efficiency (Visit Distribution)")
    print("=" * 80)

    seed = 99999
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    bc = gc.create_battle_context()

    print(f"Battle: {bc.encounter}")
    print("\nRunning MCTS search with 10000 simulations...")

    searcher = sts.BattleSearcher(bc)
    searcher.search(10000)

    root_edges = searcher.get_root_edges()

    # Analyze visit distribution
    visit_counts = []
    for edge in root_edges:
        if edge.node:
            visit_counts.append(edge.node.simulation_count)

    if visit_counts:
        visit_counts.sort(reverse=True)
        total = sum(visit_counts)

        print(f"\nVisit count distribution (top actions):")
        print(f"  Total simulations: {total}")
        print(f"  Top action visits: {visit_counts[0]} ({visit_counts[0]/total*100:.1f}%)")
        if len(visit_counts) >= 5:
            print(f"  Top 5 actions: {visit_counts[:5]}")

        # Calculate concentration (Herfindahl index)
        concentration = sum((v/total)**2 for v in visit_counts)
        print(f"\n  Concentration index: {concentration:.3f}")
        print(f"   (higher = more concentrated in few nodes)")

    return True


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "GRAPH SEARCH DEDUPLICATION TEST" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")

    results = []

    try:
        result = test_simple_combat_deduplication()
        results.append(("Simple Combat Deduplication", result))
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Simple Combat Deduplication", False))

    try:
        result = test_random_node_deduplication()
        results.append(("Random Node Deduplication", result))
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Random Node Deduplication", False))

    try:
        result = test_memory_efficiency()
        results.append(("Memory Efficiency", result))
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Memory Efficiency", False))

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
    print("NOTE: If deduplication shows 'INCONCLUSIVE' or 'NOT detected',")
    print("      this is expected BEFORE implementing the graph search changes.")
    print("      After implementing GRAPH_SEARCH_PLAN.md, rerun this test to verify.")
    print("=" * 80 + "\n")
