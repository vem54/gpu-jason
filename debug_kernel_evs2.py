"""
Debug: Print EVs for all nodes in the tree.
"""

import numpy as np
import cupy as cp
from gpu_poker_cfr.games.turn_toy_game import make_turn_toy_game, build_turn_tree, print_tree_structure
from gpu_poker_cfr.games.turn_toy_game import NODE_FOLD, NODE_SHOWDOWN, NODE_DECISION, NODE_CHANCE
from gpu_poker_cfr.games.cards import card_name

# Reuse the debug kernel from before
from debug_kernel_evs import DEBUG_KERNEL


def run_debug():
    print("=" * 60)
    print("DEBUGGING GPU KERNEL EVs - ALL NODES")
    print("=" * 60)

    game = make_turn_toy_game()
    nodes, child_indices = build_turn_tree(game)
    num_nodes = len(nodes)
    num_deals = game.num_deals
    max_actions = 2

    print("\nTree structure:")
    print_tree_structure(nodes, child_indices)

    from gpu_poker_cfr.solvers.gpu_turn_solver import GPUTurnSolver
    solver = GPUTurnSolver(game)

    # Initialize uniform strategy
    uniform_strategy = cp.ones((solver.num_infosets, max_actions), dtype=cp.float32) / max_actions

    # Allocate debug arrays
    debug_ev0 = cp.zeros((num_nodes, num_deals), dtype=cp.float32)
    debug_ev1 = cp.zeros((num_nodes, num_deals), dtype=cp.float32)
    debug_chance_ev0 = cp.zeros((8, num_deals), dtype=cp.float32)
    debug_chance_ev1 = cp.zeros((8, num_deals), dtype=cp.float32)
    debug_num_rivers = cp.zeros(num_deals, dtype=cp.int32)

    # Run debug kernel
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
         uniform_strategy,
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

    print("\n--- EVs for ALL nodes (IP perspective) ---")
    type_names = {0: 'FOLD', 1: 'SHOW', 2: 'DEC', 3: 'CHNC'}
    for i, node in enumerate(nodes):
        t = type_names[node['type']]
        player = node.get('player', '-')
        children = child_indices[i] if i < len(child_indices) else [-1, -1]
        is_river = solver.node_is_river[i]

        print(f"  [{i:2d}] {t} P{player} {'R' if is_river else 'T'} "
              f"pot={node['pot']:.0f} ev1={ev1[i, d]:7.2f} -> children {list(children)}")

    # Print invested amounts for key nodes
    print("\n--- Invested amounts for key nodes ---")
    for i in [5, 6, 8, 9, 10, 13, 18, 20, 22, 23]:
        if i < num_nodes:
            node = nodes[i]
            inv_oop = node.get('invested_oop', 0)
            inv_ip = node.get('invested_ip', 0)
            print(f"  Node {i}: invested_oop={inv_oop}, invested_ip={inv_ip}")

    # Trace the all-in call path: Node 4 -> Node 9 -> Node 13 -> Node 19 -> Node 23
    print("\n--- Tracing all-in call path (IP JJ vs OOP AA) ---")
    print("  Node 4 (OOP decision): All-in child = Node 9 (CHANCE)")
    print(f"    EV at Node 4 = {ev1[4, d]:.2f}")
    print(f"  Node 9 (CHANCE): child = Node 13")
    print(f"    chance_ev1[2] = {chance_ev1[2, d]:.2f}")
    print(f"    ev1[9] = {ev1[9, d]:.2f}")
    print(f"  Node 13 (OOP river): child = Node 19")
    print(f"    ev1[13] = {ev1[13, d]:.2f}")
    print(f"  Node 19 (IP river): child = Node 23")
    print(f"    ev1[19] = {ev1[19, d]:.2f}")
    print(f"  Node 23 (SHOWDOWN):")
    print(f"    ev1[23] = {ev1[23, d]:.2f}")

    # Manually compute showdown EV at Node 23
    print("\n--- Manual showdown EV at Node 23 ---")
    valid_rivers = game.get_valid_rivers(d)
    from gpu_poker_cfr.games.hand_eval import evaluate_7cards

    showdown_evs = []
    for river in valid_rivers:
        board = list(game.board) + [river]
        h0 = np.array(list(oop_hand) + board, dtype=np.int32)
        h1 = np.array(list(ip_hand) + board, dtype=np.int32)
        v0, v1 = evaluate_7cards(h0), evaluate_7cards(h1)

        # Node 23 is at pot=300, invested_oop=150, invested_ip=150
        inv_oop = 150
        inv_ip = 150
        if v1 > v0:  # IP wins
            ev = inv_oop  # IP gains OOP's investment
        elif v0 > v1:  # OOP wins
            ev = -inv_ip  # IP loses their investment
        else:
            ev = 0

        showdown_evs.append(ev)

    avg_showdown = np.mean(showdown_evs)
    print(f"  Average showdown EV for IP: {avg_showdown:.2f}")
    print(f"  (vs GPU ev1[23] = {ev1[23, d]:.2f} - this is for ONE river only)")


if __name__ == '__main__':
    run_debug()
