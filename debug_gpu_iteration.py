"""
Debug: Trace GPU CFR iteration to check EV computation.
We'll run a single iteration and print the internal state.
"""

import numpy as np
import cupy as cp
from gpu_poker_cfr.games.turn_toy_game import make_turn_toy_game
from gpu_poker_cfr.solvers.gpu_turn_solver import GPUTurnSolver
from gpu_poker_cfr.games.cards import card_name


def debug_iteration():
    print("=" * 60)
    print("DEBUG GPU CFR ITERATION")
    print("=" * 60)

    game = make_turn_toy_game()
    solver = GPUTurnSolver(game)

    print("\n--- Initial state ---")
    print(f"Cumulative regret shape: {solver._cumulative_regret.shape}")
    print(f"Strategy shape: {solver._strategy.shape}")

    # Run 1 iteration
    print("\n--- Running 1 iteration ---")
    solver.iterate(1)

    # Get regrets for Node 0 (OOP turn)
    print("\n--- After 1 iteration ---")

    # Find infoset for OOP AA at Node 0
    oop_aa_hands = [h for h in solver.oop_hands if h[0] // 4 == 12 and h[1] // 4 == 12]
    print(f"OOP AA hands: {[tuple(card_name(c) for c in h) for h in oop_aa_hands]}")

    if oop_aa_hands:
        aa_hand = oop_aa_hands[0]
        hand_idx = solver.oop_hand_to_idx[aa_hand]
        key = (0, hand_idx, 0)  # player 0, hand_idx, node 0

        if key in solver.infoset_key_to_idx:
            infoset_idx = solver.infoset_key_to_idx[key]
            regrets = cp.asnumpy(solver._cumulative_regret[infoset_idx])
            print(f"\nOOP {card_name(aa_hand[0])}{card_name(aa_hand[1])} at Node 0:")
            print(f"  Infoset idx: {infoset_idx}")
            print(f"  Cumulative regrets: Check={regrets[0]:.2f}, All-in={regrets[1]:.2f}")

            strat_sum = cp.asnumpy(solver._cumulative_strategy[infoset_idx])
            print(f"  Cumulative strategy: Check={strat_sum[0]:.2f}, All-in={strat_sum[1]:.2f}")

    # Run more iterations
    print("\n--- Running 99 more iterations (100 total) ---")
    solver.iterate(99)

    if oop_aa_hands:
        aa_hand = oop_aa_hands[0]
        hand_idx = solver.oop_hand_to_idx[aa_hand]
        key = (0, hand_idx, 0)

        if key in solver.infoset_key_to_idx:
            infoset_idx = solver.infoset_key_to_idx[key]
            regrets = cp.asnumpy(solver._cumulative_regret[infoset_idx])
            print(f"\nOOP {card_name(aa_hand[0])}{card_name(aa_hand[1])} at Node 0:")
            print(f"  Cumulative regrets: Check={regrets[0]:.2f}, All-in={regrets[1]:.2f}")

            strat_sum = cp.asnumpy(solver._cumulative_strategy[infoset_idx])
            total = strat_sum[0] + strat_sum[1]
            if total > 0:
                probs = strat_sum / total
                print(f"  Average strategy: Check={probs[0]*100:.1f}%, All-in={probs[1]*100:.1f}%")

    # Check aggregate strategy at Node 0
    print("\n--- Aggregate strategy at Node 0 ---")
    agg = solver.get_aggregate_strategy(0)
    print(f"  {agg}")

    # Run 1000 total iterations
    print("\n--- Running to 1000 iterations ---")
    solver.iterate(900)

    agg = solver.get_aggregate_strategy(0)
    print(f"Aggregate at Node 0 (1000 iter): {agg}")

    agg = solver.get_aggregate_strategy(1)
    print(f"Aggregate at Node 1 (1000 iter): {agg}")

    print("\n--- Compare to CPU ---")
    print("Running CPU solver...")
    from cpu_turn_cfr_reference import CPUTurnCFR
    cpu_solver = CPUTurnCFR(game)
    cpu_solver.iterate(100)

    print("\nCPU at Node 0 (100 iter):", cpu_solver.get_aggregate_strategy(0))
    print("GPU at Node 0 (1000 iter):", solver.get_aggregate_strategy(0))


if __name__ == '__main__':
    debug_iteration()
