"""
Debug: Use actual solver strategy for EV computation.
"""

import numpy as np
import cupy as cp
from gpu_poker_cfr.games.turn_toy_game import make_turn_toy_game, build_turn_tree
from gpu_poker_cfr.games.turn_toy_game import NODE_FOLD, NODE_SHOWDOWN, NODE_DECISION, NODE_CHANCE
from gpu_poker_cfr.games.cards import card_name
from debug_kernel_evs import DEBUG_KERNEL


def run_debug():
    print("=" * 60)
    print("DEBUGGING GPU KERNEL EVs - CORRECT STRATEGY")
    print("=" * 60)

    game = make_turn_toy_game()
    nodes, child_indices = build_turn_tree(game)
    num_nodes = len(nodes)
    num_deals = game.num_deals
    max_actions = 2

    from gpu_poker_cfr.solvers.gpu_turn_solver import GPUTurnSolver
    solver = GPUTurnSolver(game)

    # Run 1 iteration to get proper strategy
    solver.iterate(1)

    # Get the computed strategy (with correct single-action handling)
    strategy = solver._strategy

    # Allocate debug arrays
    debug_ev0 = cp.zeros((num_nodes, num_deals), dtype=cp.float32)
    debug_ev1 = cp.zeros((num_nodes, num_deals), dtype=cp.float32)
    debug_chance_ev0 = cp.zeros((8, num_deals), dtype=cp.float32)
    debug_chance_ev1 = cp.zeros((8, num_deals), dtype=cp.float32)
    debug_num_rivers = cp.zeros(num_deals, dtype=cp.int32)

    # Run debug kernel with correct strategy
    block_size = 128
    grid_size = (num_deals + block_size - 1) // block_size
    num_possible_rivers = len(game.possible_rivers)

    DEBUG_KERNEL(
        (grid_size,), (block_size,),
        (np.int32(num_nodes),
         np.int32(max_actions),
         solver.node_type_gpu,
         solver.node_player_gpu,
         solver.node_num_actions_gpu,
         solver.node_fold_player_gpu,
         solver.node_children_gpu,
         solver.node_invested_oop_gpu,
         solver.node_invested_ip_gpu,
         solver.node_is_river_gpu,
         np.int32(num_deals),
         np.int32(num_possible_rivers),
         solver.deal_river_valid_gpu,
         solver.deal_river_p0_value_gpu,
         solver.deal_river_p1_value_gpu,
         solver.node_h_p0,
         solver.node_h_p1,
         strategy,  # Use actual strategy
         np.int32(solver.num_infosets),
         debug_ev0,
         debug_ev1,
         debug_chance_ev0,
         debug_chance_ev1,
         debug_num_rivers)
    )

    cp.cuda.Stream.null.synchronize()

    # Convert to numpy
    ev0 = cp.asnumpy(debug_ev0)
    ev1 = cp.asnumpy(debug_ev1)
    chance_ev0 = cp.asnumpy(debug_chance_ev0)
    chance_ev1 = cp.asnumpy(debug_chance_ev1)
    num_rivers = cp.asnumpy(debug_num_rivers)

    # Find JJ deal
    jj_deal = None
    for d in range(num_deals):
        oop_hand, ip_hand = game.get_deal(d)
        if ip_hand[0] // 4 == 9 and ip_hand[1] // 4 == 9:  # JJ
            jj_deal = d
            break

    if jj_deal is None:
        print("No JJ deal found")
        return

    d = jj_deal
    oop_hand, ip_hand = game.get_deal(d)
    print(f"\n=== Deal {d}: OOP={card_name(oop_hand[0])}{card_name(oop_hand[1])}, IP={card_name(ip_hand[0])}{card_name(ip_hand[1])} ===")
    print(f"Valid rivers: {num_rivers[d]}")

    print("\n--- Key Node EVs for IP ---")
    print(f"  Node 1 (IP turn after OOP check):")
    print(f"    EV = {ev1[1, d]:.2f}")
    print(f"    Child 3 (CHANCE - check): EV = {ev1[3, d]:.2f}")
    print(f"    Child 4 (OOP faces all-in): EV = {ev1[4, d]:.2f}")

    print(f"\n  Node 4 (OOP facing all-in):")
    print(f"    EV = {ev1[4, d]:.2f}")
    print(f"    Child 8 (FOLD): EV = {ev1[8, d]:.2f}")
    print(f"    Child 9 (CHANCE - call): EV = {ev1[9, d]:.2f}")

    print(f"\n  Node 9 (CHANCE after all-in call):")
    print(f"    chance_ev1 = {chance_ev1[2, d]:.2f}")
    print(f"    Child 13 EV = {ev1[13, d]:.2f}")
    print(f"    Node 19 EV = {ev1[19, d]:.2f}")
    print(f"    Node 23 (SHOWDOWN) EV = {ev1[23, d]:.2f}")

    # Manual calculation
    print("\n--- Expected values (manual, averaged) ---")
    valid_rivers = game.get_valid_rivers(d)
    from gpu_poker_cfr.games.hand_eval import evaluate_7cards

    check_ev = allin_call_ev = 0
    for river in valid_rivers:
        board = list(game.board) + [river]
        h0 = np.array(list(oop_hand) + board, dtype=np.int32)
        h1 = np.array(list(ip_hand) + board, dtype=np.int32)
        v0, v1 = evaluate_7cards(h0), evaluate_7cards(h1)

        if v1 > v0:
            check_ev += 50
            allin_call_ev += 150
        elif v0 > v1:
            check_ev -= 50
            allin_call_ev -= 150

    n = len(valid_rivers)
    print(f"  Check-check showdown (avg): {check_ev / n:.2f}")
    print(f"  All-in call showdown (avg): {allin_call_ev / n:.2f}")
    print(f"  All-in (50/50 OOP): {0.5 * 50 + 0.5 * allin_call_ev / n:.2f}")

    # Compare
    print("\n--- Comparison ---")
    print(f"  GPU Node 9 CHANCE EV: {chance_ev1[2, d]:.2f}")
    print(f"  Expected avg showdown: {allin_call_ev / n:.2f}")
    print(f"  Ratio: {chance_ev1[2, d] / (allin_call_ev / n):.3f}")


if __name__ == '__main__':
    run_debug()
