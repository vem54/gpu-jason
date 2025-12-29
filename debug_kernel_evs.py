"""
Debug: Capture actual EVs from the GPU kernel by modifying it to store debug info.
"""

import numpy as np
import cupy as cp
from gpu_poker_cfr.games.turn_toy_game import make_turn_toy_game, build_turn_tree
from gpu_poker_cfr.games.turn_toy_game import NODE_FOLD, NODE_SHOWDOWN, NODE_DECISION, NODE_CHANCE
from gpu_poker_cfr.games.cards import card_name


# Modified kernel with debug output
DEBUG_KERNEL = cp.RawKernel(r'''
extern "C" __global__ void debug_turn_cfr(
    const int num_nodes,
    const int max_actions,
    const int* __restrict__ node_type,
    const int* __restrict__ node_player,
    const int* __restrict__ node_num_actions,
    const int* __restrict__ node_fold_player,
    const int* __restrict__ node_children,
    const float* __restrict__ node_invested_oop,
    const float* __restrict__ node_invested_ip,
    const int* __restrict__ node_is_river,

    const int num_deals,
    const int num_possible_rivers,
    const int* __restrict__ deal_river_valid,
    const int* __restrict__ deal_river_p0_value,
    const int* __restrict__ deal_river_p1_value,

    const int* __restrict__ node_h_p0,
    const int* __restrict__ node_h_p1,
    const float* __restrict__ strategy,
    const int num_infosets,

    // Debug output arrays
    float* __restrict__ debug_ev0,  // [num_nodes * num_deals]
    float* __restrict__ debug_ev1,
    float* __restrict__ debug_chance_ev0,  // [8 * num_deals]
    float* __restrict__ debug_chance_ev1,
    int* __restrict__ debug_num_rivers  // [num_deals]
) {
    int deal = blockIdx.x * blockDim.x + threadIdx.x;
    if (deal >= num_deals) return;

    int num_valid_rivers = 0;
    for (int r = 0; r < num_possible_rivers; r++) {
        if (deal_river_valid[deal * num_possible_rivers + r]) {
            num_valid_rivers++;
        }
    }
    if (num_valid_rivers == 0) return;

    debug_num_rivers[deal] = num_valid_rivers;
    float river_weight = 1.0f / num_valid_rivers;

    float ev0[64];
    float ev1[64];
    float reach0[64];
    float reach1[64];

    for (int i = 0; i < 64; i++) {
        ev0[i] = 0.0f;
        ev1[i] = 0.0f;
        reach0[i] = 0.0f;
        reach1[i] = 0.0f;
    }

    reach0[0] = 1.0f;
    reach1[0] = 1.0f;

    // Forward pass
    for (int i = 0; i < num_nodes; i++) {
        int ntype = node_type[i];

        if (ntype == 3) {
            int child = node_children[i * max_actions + 0];
            if (child >= 0 && child < 64) {
                reach0[child] = reach0[i];
                reach1[child] = reach1[i];
            }
            continue;
        }

        if (ntype != 2) continue;

        int player = node_player[i];
        int n_act = node_num_actions[i];

        int h_idx;
        if (player == 0) {
            h_idx = node_h_p0[i * num_deals + deal];
        } else {
            h_idx = node_h_p1[i * num_deals + deal];
        }

        if (h_idx < 0) {
            for (int a = 0; a < n_act; a++) {
                int child = node_children[i * max_actions + a];
                if (child >= 0 && child < 64) {
                    reach0[child] = 0.0f;
                    reach1[child] = 0.0f;
                }
            }
            continue;
        }

        int strat_base = h_idx * max_actions;

        for (int a = 0; a < n_act; a++) {
            float s = strategy[strat_base + a];
            int child = node_children[i * max_actions + a];
            if (child >= 0 && child < 64) {
                if (player == 0) {
                    reach0[child] = reach0[i] * s;
                    reach1[child] = reach1[i];
                } else {
                    reach0[child] = reach0[i];
                    reach1[child] = reach1[i] * s;
                }
            }
        }
    }

    // Chance node EVs
    float chance_ev0[8];
    float chance_ev1[8];
    int chance_node_ids[8];
    int num_chance_nodes = 0;

    for (int i = 0; i < num_nodes && num_chance_nodes < 8; i++) {
        if (node_type[i] == 3) {
            chance_node_ids[num_chance_nodes] = i;
            chance_ev0[num_chance_nodes] = 0.0f;
            chance_ev1[num_chance_nodes] = 0.0f;
            num_chance_nodes++;
        }
    }

    // River backward pass
    for (int r = 0; r < num_possible_rivers; r++) {
        if (!deal_river_valid[deal * num_possible_rivers + r]) continue;

        int p0_val = deal_river_p0_value[deal * num_possible_rivers + r];
        int p1_val = deal_river_p1_value[deal * num_possible_rivers + r];

        for (int i = num_nodes - 1; i >= 0; i--) {
            if (!node_is_river[i]) continue;

            int ntype = node_type[i];

            if (ntype == 0) {  // FOLD
                int fp = node_fold_player[i];
                float inv_oop = node_invested_oop[i];
                float inv_ip = node_invested_ip[i];
                if (fp == 0) {
                    ev0[i] = -inv_oop;
                    ev1[i] = inv_oop;
                } else {
                    ev0[i] = inv_ip;
                    ev1[i] = -inv_ip;
                }
            }
            else if (ntype == 1) {  // SHOWDOWN
                float inv_oop = node_invested_oop[i];
                float inv_ip = node_invested_ip[i];
                if (p0_val > p1_val) {
                    ev0[i] = inv_ip;
                    ev1[i] = -inv_ip;
                } else if (p1_val > p0_val) {
                    ev0[i] = -inv_oop;
                    ev1[i] = inv_oop;
                } else {
                    ev0[i] = 0.0f;
                    ev1[i] = 0.0f;
                }
            }
            else if (ntype == 2) {  // DECISION
                int player = node_player[i];
                int n_act = node_num_actions[i];

                int h_idx;
                if (player == 0) {
                    h_idx = node_h_p0[i * num_deals + deal];
                } else {
                    h_idx = node_h_p1[i * num_deals + deal];
                }

                if (h_idx < 0) continue;

                int strat_base = h_idx * max_actions;

                float node_ev0 = 0.0f;
                float node_ev1 = 0.0f;

                for (int a = 0; a < n_act; a++) {
                    int child = node_children[i * max_actions + a];
                    float s = strategy[strat_base + a];
                    if (child >= 0 && child < 64) {
                        node_ev0 += s * ev0[child];
                        node_ev1 += s * ev1[child];
                    }
                }

                ev0[i] = node_ev0;
                ev1[i] = node_ev1;
            }
        }

        // Accumulate chance EVs
        for (int c = 0; c < num_chance_nodes; c++) {
            int chance_idx = chance_node_ids[c];
            int child = node_children[chance_idx * max_actions + 0];
            if (child >= 0 && child < 64) {
                chance_ev0[c] += ev0[child] * river_weight;
                chance_ev1[c] += ev1[child] * river_weight;
            }
        }
    }

    // Store chance EVs
    for (int c = 0; c < num_chance_nodes; c++) {
        debug_chance_ev0[c * num_deals + deal] = chance_ev0[c];
        debug_chance_ev1[c * num_deals + deal] = chance_ev1[c];
    }

    // Turn backward pass
    for (int i = num_nodes - 1; i >= 0; i--) {
        if (node_is_river[i]) continue;

        int ntype = node_type[i];

        if (ntype == 0) {  // FOLD
            int fp = node_fold_player[i];
            float inv_oop = node_invested_oop[i];
            float inv_ip = node_invested_ip[i];
            if (fp == 0) {
                ev0[i] = -inv_oop;
                ev1[i] = inv_oop;
            } else {
                ev0[i] = inv_ip;
                ev1[i] = -inv_ip;
            }
        }
        else if (ntype == 3) {  // CHANCE
            for (int c = 0; c < num_chance_nodes; c++) {
                if (chance_node_ids[c] == i) {
                    ev0[i] = chance_ev0[c];
                    ev1[i] = chance_ev1[c];
                    break;
                }
            }
        }
        else if (ntype == 2) {  // DECISION
            int player = node_player[i];
            int n_act = node_num_actions[i];

            int h_idx;
            if (player == 0) {
                h_idx = node_h_p0[i * num_deals + deal];
            } else {
                h_idx = node_h_p1[i * num_deals + deal];
            }

            if (h_idx < 0) continue;

            int strat_base = h_idx * max_actions;

            float node_ev0 = 0.0f;
            float node_ev1 = 0.0f;

            for (int a = 0; a < n_act; a++) {
                int child = node_children[i * max_actions + a];
                float s = strategy[strat_base + a];
                if (child >= 0 && child < 64) {
                    node_ev0 += s * ev0[child];
                    node_ev1 += s * ev1[child];
                }
            }

            ev0[i] = node_ev0;
            ev1[i] = node_ev1;
        }
    }

    // Store all EVs
    for (int i = 0; i < num_nodes; i++) {
        debug_ev0[i * num_deals + deal] = ev0[i];
        debug_ev1[i * num_deals + deal] = ev1[i];
    }
}
''', 'debug_turn_cfr')


def run_debug():
    print("=" * 60)
    print("DEBUGGING GPU KERNEL EVs")
    print("=" * 60)

    game = make_turn_toy_game()
    nodes, child_indices = build_turn_tree(game)
    num_nodes = len(nodes)
    num_deals = game.num_deals
    max_actions = 2

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

    # Find a JJ deal
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
    print(f"\nDeal {d}: OOP={card_name(oop_hand[0])}{card_name(oop_hand[1])}, IP={card_name(ip_hand[0])}{card_name(ip_hand[1])}")
    print(f"Valid rivers: {num_rivers[d]}")

    print("\n--- Node 1 (IP turn after OOP check) ---")
    print(f"  Node 1 EV for IP: {ev1[1, d]:.2f}")
    print(f"  Child 3 (CHANCE) EV for IP: {ev1[3, d]:.2f}")
    print(f"  Child 4 (OOP decision) EV for IP: {ev1[4, d]:.2f}")

    print("\n--- Node 4 (OOP decision facing all-in) ---")
    print(f"  Node 4 EV for IP: {ev1[4, d]:.2f}")
    print(f"  Child 8 (FOLD) EV for IP: {ev1[8, d]:.2f}")
    print(f"  Child 9 (CHANCE) EV for IP: {ev1[9, d]:.2f}")

    print("\n--- Chance node accumulated EVs ---")
    print(f"  Node 3 (CHANCE): IP EV = {chance_ev1[0, d]:.2f}")
    print(f"  Node 6 (CHANCE): IP EV = {chance_ev1[1, d]:.2f}")
    print(f"  Node 9 (CHANCE): IP EV = {chance_ev1[2, d]:.2f}")

    # Expected values (manual calculation)
    print("\n--- Expected values (manual calculation) ---")
    valid_rivers = game.get_valid_rivers(d)
    n = len(valid_rivers)

    check_ev = 0
    allin_call_ev = 0
    for river in valid_rivers:
        from gpu_poker_cfr.games.hand_eval import evaluate_7cards
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

    print(f"  Check-check showdown (avg): {check_ev / n:.2f}")
    print(f"  All-in call showdown (avg): {allin_call_ev / n:.2f}")
    print(f"  All-in + OOP folds: {50}")
    print(f"  All-in (50/50 OOP): {0.5 * 50 + 0.5 * allin_call_ev / n:.2f}")


if __name__ == '__main__':
    run_debug()
