"""
Debug: Compare river node regret accumulation between CPU and GPU.

Hypothesis: GPU multiplies river regrets by river_weight (1/44) but CPU doesn't.
This causes a 44x scale difference in river node regrets.
"""

import numpy as np
import cupy as cp
from gpu_poker_cfr.games.turn_toy_game import make_turn_toy_game
from gpu_poker_cfr.games.turn_toy_game import NODE_FOLD, NODE_SHOWDOWN, NODE_DECISION, NODE_CHANCE
from gpu_poker_cfr.games.cards import card_name
from cpu_turn_cfr_reference import CPUTurnCFR


def compare_regrets_after_one_iteration():
    """Compare regrets after a single iteration."""
    print("=" * 60)
    print("COMPARING CPU vs GPU REGRETS AFTER 1 ITERATION")
    print("=" * 60)

    game = make_turn_toy_game()

    # Run CPU solver for 1 iteration
    print("\nRunning CPU CFR (1 iteration)...")
    cpu_solver = CPUTurnCFR(game)
    cpu_solver.iterate(1)

    # Run GPU solver for 1 iteration
    print("Running GPU CFR (1 iteration)...")
    from gpu_poker_cfr.solvers.gpu_turn_solver import GPUTurnSolver
    gpu_solver = GPUTurnSolver(game)
    gpu_solver.iterate(1)

    # Find IP JJ infoset at Node 1 (turn node)
    ip_jj = None
    for hand in cpu_solver.ip_hands:
        if hand[0] // 4 == 9 and hand[1] // 4 == 9:
            ip_jj = hand
            break

    print(f"\nIP JJ hand: {card_name(ip_jj[0])}{card_name(ip_jj[1])}")

    # Node 1 is IP turn after OOP check
    node_idx = 1
    player = 1

    # Get CPU regrets
    cpu_key = (player, ip_jj, node_idx)
    cpu_regrets = cpu_solver.cumulative_regret.get(cpu_key, [0.0, 0.0])
    print(f"\nNode 1 (IP turn - TURN node) regrets for IP JJ:")
    print(f"  CPU: Check={cpu_regrets[0]:.2f}, All-in={cpu_regrets[1]:.2f}")

    # Get GPU regrets
    ip_jj_idx = gpu_solver.ip_hand_to_idx[ip_jj]
    gpu_key = (player, ip_jj_idx, node_idx)
    h_idx = gpu_solver.infoset_key_to_idx.get(gpu_key)
    if h_idx is not None:
        gpu_regrets = cp.asnumpy(gpu_solver._cumulative_regret[h_idx])
        print(f"  GPU: Check={gpu_regrets[0]:.2f}, All-in={gpu_regrets[1]:.2f}")

        ratio_check = gpu_regrets[0]/cpu_regrets[0] if cpu_regrets[0] != 0 else float('inf')
        ratio_allin = gpu_regrets[1]/cpu_regrets[1] if cpu_regrets[1] != 0 else float('inf')
        print(f"  Ratio (GPU/CPU): Check={ratio_check:.4f}, All-in={ratio_allin:.4f}")

        # Check if they have opposite signs (one is 0 and other is positive)
        if (cpu_regrets[0] > 0 and gpu_regrets[1] > 0) or (cpu_regrets[1] > 0 and gpu_regrets[0] > 0):
            print("\n  *** WARNING: CPU and GPU have OPPOSITE regret preferences! ***")

    # Now check a river node - let's find one
    # Node 10 should be a river node (OOP decision after check-check-check)
    print("\n--- River Node Comparison ---")

    # Find the river nodes in the tree
    river_decision_nodes = []
    for i, node in enumerate(gpu_solver.nodes):
        if node['type'] == NODE_DECISION and gpu_solver.node_is_river[i]:
            river_decision_nodes.append(i)

    print(f"River decision nodes: {river_decision_nodes}")

    if river_decision_nodes:
        river_node = river_decision_nodes[0]
        river_player = gpu_solver.nodes[river_node]['player']
        print(f"\nRiver node {river_node} (player {river_player}):")

        if river_player == 0:  # OOP
            test_hand = cpu_solver.oop_hands[0]
            hand_idx = gpu_solver.oop_hand_to_idx[test_hand]
        else:  # IP
            test_hand = cpu_solver.ip_hands[0]
            hand_idx = gpu_solver.ip_hand_to_idx[test_hand]

        print(f"  Testing hand: {card_name(test_hand[0])}{card_name(test_hand[1])}")

        # CPU regrets
        cpu_key_river = (river_player, test_hand, river_node)
        cpu_regrets_river = cpu_solver.cumulative_regret.get(cpu_key_river, [0.0, 0.0])
        print(f"  CPU regrets: {cpu_regrets_river}")

        # GPU regrets
        gpu_key_river = (river_player, hand_idx, river_node)
        h_idx_river = gpu_solver.infoset_key_to_idx.get(gpu_key_river)
        if h_idx_river is not None:
            gpu_regrets_river = cp.asnumpy(gpu_solver._cumulative_regret[h_idx_river])
            print(f"  GPU regrets: {gpu_regrets_river[:2]}")

            if cpu_regrets_river[0] != 0:
                ratio = gpu_regrets_river[0] / cpu_regrets_river[0]
                print(f"  Ratio (GPU/CPU): {ratio:.4f}")

                # Expected ratio should be 1/44 if bug exists
                n_rivers = 44  # approximate
                print(f"  Expected if bug exists: ~{1/n_rivers:.4f}")


def analyze_river_regret_bug():
    """Analyze the potential river regret bug in detail."""
    print("\n" + "=" * 60)
    print("RIVER REGRET SCALE ANALYSIS")
    print("=" * 60)

    game = make_turn_toy_game()

    # Count average valid rivers per deal
    total_rivers = 0
    for d in range(game.num_deals):
        total_rivers += len(game.get_valid_rivers(d))
    avg_rivers = total_rivers / game.num_deals
    print(f"\nAverage valid rivers per deal: {avg_rivers:.1f}")

    # In GPU kernel, river regrets are multiplied by 1/num_valid_rivers
    # In CPU, river regrets are added once per river
    # So GPU river regrets should be ~1/44 of CPU river regrets

    print(f"\nExpected GPU/CPU ratio for river nodes: ~{1/avg_rivers:.4f}")

    # But turn nodes don't have this issue - they use averaged EVs from chance nodes
    print(f"Expected GPU/CPU ratio for turn nodes: ~1.0")

    print("\n--- Impact on Equilibrium ---")
    print("If river node regrets are 44x smaller than expected:")
    print("  - River node strategies will change 44x slower")
    print("  - OOP at Node 4 (facing all-in) won't adapt properly")
    print("  - This could cause IP to over-value/under-value all-in")


if __name__ == '__main__':
    compare_regrets_after_one_iteration()
    analyze_river_regret_bug()
