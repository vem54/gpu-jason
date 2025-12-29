"""Test script for GPU turn solver."""

import sys
import time

print("Importing modules...")
sys.stdout.flush()

from gpu_poker_cfr.games.turn_toy_game import make_turn_toy_game, build_turn_tree, print_tree_structure
from gpu_poker_cfr.solvers.gpu_turn_solver import GPUTurnSolver

def test_turn_tree():
    """Test that turn tree builds correctly."""
    print("\n=== Testing Turn Tree ===")
    game = make_turn_toy_game()
    nodes, child_indices = build_turn_tree(game)
    print_tree_structure(nodes, child_indices)

    # Count node types
    type_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for node in nodes:
        type_counts[node['type']] += 1

    print(f"\nNode type counts:")
    print(f"  FOLD: {type_counts[0]}")
    print(f"  SHOWDOWN: {type_counts[1]}")
    print(f"  DECISION: {type_counts[2]}")
    print(f"  CHANCE: {type_counts[3]}")

    return game

def test_turn_solver(game, iterations=1000):
    """Test the GPU turn solver."""
    print(f"\n=== Testing GPU Turn Solver ({iterations} iterations) ===")

    print("Creating solver...")
    start = time.time()
    solver = GPUTurnSolver(game)
    init_time = time.time() - start
    print(f"Initialization: {init_time:.2f}s")

    print("Running CFR iterations...")
    start = time.time()
    solver.solve(iterations=iterations)
    solve_time = time.time() - start
    print(f"Solving: {solve_time:.2f}s ({iterations/solve_time:.0f} iter/s)")

    # Print strategies
    print("\n=== Turn Root Strategy (OOP, node 0) ===")
    root_strat = solver.get_aggregate_strategy(0)
    for action, prob in root_strat.items():
        print(f"  {action}: {prob*100:.1f}%")

    # Find IP response node (after OOP check)
    for node_idx, node in enumerate(solver.nodes):
        if node['type'] == 2 and node['player'] == 1:
            street = node.get('street', 'unknown')
            to_call = node.get('to_call', 0)

            if street == 'turn':
                strat = solver.get_aggregate_strategy(node_idx)
                if strat:
                    print(f"\n=== Node {node_idx}: IP {street} (to_call={to_call}) ===")
                    for action, prob in strat.items():
                        print(f"  {action}: {prob*100:.1f}%")

    return solver

def main():
    print("=" * 60)
    print("GPU Turn Solver Test")
    print("=" * 60)

    # Test tree building
    game = test_turn_tree()

    # Test solver with small number of iterations first
    print("\n--- Quick test (100 iterations) ---")
    solver = test_turn_solver(game, iterations=100)

    # Then with more iterations
    print("\n--- Full test (10000 iterations) ---")
    solver = test_turn_solver(game, iterations=10000)

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
