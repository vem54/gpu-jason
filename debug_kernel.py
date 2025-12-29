"""Debug the CUDA kernel."""

import numpy as np
import cupy as cp
from gpu_poker_cfr.games.river_toy import RiverMicro
from gpu_poker_cfr.games.hand_eval import evaluate_7cards


# Simple debug kernel that prints values
DEBUG_KERNEL = cp.RawKernel(r'''
extern "C" __global__ void debug_cfr(
    const int num_nodes,
    const int* __restrict__ node_type,
    const int* __restrict__ node_player,
    const float* __restrict__ node_pot,
    const int* __restrict__ node_num_actions,
    const int* __restrict__ node_fold_player,
    const int* __restrict__ node_child_idx,

    const int num_deals,
    const int* __restrict__ deal_p0_value,
    const int* __restrict__ deal_p1_value,

    const int* __restrict__ node_h_p0,
    const int* __restrict__ node_h_p1,

    const float* __restrict__ strat,
    float* __restrict__ cumulative_regret,

    const int num_infosets,
    const float regret_scale,

    // Debug output
    float* __restrict__ debug_ev0,
    float* __restrict__ debug_ev1
) {
    int deal = blockIdx.x * blockDim.x + threadIdx.x;
    if (deal >= num_deals) return;

    // Only debug first deal
    if (deal != 0) return;

    float ev0[32];
    float ev1[32];

    // Initialize to zero for debugging
    for (int i = 0; i < 32; i++) {
        ev0[i] = 0.0f;
        ev1[i] = 0.0f;
    }

    int p0_val = deal_p0_value[deal];
    int p1_val = deal_p1_value[deal];

    printf("Deal 0: p0_val=%d, p1_val=%d, regret_scale=%f\n", p0_val, p1_val, regret_scale);

    for (int i = num_nodes - 1; i >= 0; i--) {
        int ntype = node_type[i];
        float pot = node_pot[i];
        float half_pot = pot * 0.5f;

        if (ntype == 0) {
            int fp = node_fold_player[i];
            if (fp == 0) {
                ev0[i] = -half_pot;
                ev1[i] = half_pot;
            } else {
                ev0[i] = half_pot;
                ev1[i] = -half_pot;
            }
            printf("Node %d FOLD: fp=%d, ev0=%.2f, ev1=%.2f\n", i, fp, ev0[i], ev1[i]);
        }
        else if (ntype == 1) {
            if (p0_val > p1_val) {
                ev0[i] = half_pot;
                ev1[i] = -half_pot;
            } else if (p1_val > p0_val) {
                ev0[i] = -half_pot;
                ev1[i] = half_pot;
            } else {
                ev0[i] = 0.0f;
                ev1[i] = 0.0f;
            }
            printf("Node %d SHOWDOWN: pot=%.0f, ev0=%.2f, ev1=%.2f\n", i, pot, ev0[i], ev1[i]);
        }
        else {
            int acting = node_player[i];
            int n_act = node_num_actions[i];
            int h_idx = (acting == 0) ? node_h_p0[i * num_deals + deal] : node_h_p1[i * num_deals + deal];

            printf("Node %d DECISION: acting=%d, n_act=%d, h_idx=%d\n", i, acting, n_act, h_idx);

            if (h_idx < 0) continue;

            int c0 = node_child_idx[i * 3 + 0];
            int c1 = node_child_idx[i * 3 + 1];
            int c2 = node_child_idx[i * 3 + 2];

            printf("  Children: %d, %d, %d\n", c0, c1, c2);

            int strat_base = h_idx * 3 * num_deals + deal;
            float s0 = (n_act >= 1 && c0 >= 0) ? strat[strat_base] : 0.0f;
            float s1 = (n_act >= 2 && c1 >= 0) ? strat[strat_base + num_deals] : 0.0f;
            float s2 = (n_act >= 3 && c2 >= 0) ? strat[strat_base + 2 * num_deals] : 0.0f;

            printf("  Strategy: %.3f, %.3f, %.3f\n", s0, s1, s2);

            float cev0_0 = (c0 >= 0) ? ev0[c0] : 0.0f;
            float cev0_1 = (c1 >= 0) ? ev0[c1] : 0.0f;
            float cev0_2 = (c2 >= 0) ? ev0[c2] : 0.0f;
            float cev1_0 = (c0 >= 0) ? ev1[c0] : 0.0f;
            float cev1_1 = (c1 >= 0) ? ev1[c1] : 0.0f;
            float cev1_2 = (c2 >= 0) ? ev1[c2] : 0.0f;

            printf("  Child EVs (P0): %.2f, %.2f, %.2f\n", cev0_0, cev0_1, cev0_2);
            printf("  Child EVs (P1): %.2f, %.2f, %.2f\n", cev1_0, cev1_1, cev1_2);

            float node_ev0 = s0 * cev0_0 + s1 * cev0_1 + s2 * cev0_2;
            float node_ev1 = s0 * cev1_0 + s1 * cev1_1 + s2 * cev1_2;

            ev0[i] = node_ev0;
            ev1[i] = node_ev1;

            printf("  Node EV: ev0=%.2f, ev1=%.2f\n", node_ev0, node_ev1);

            // Regret updates
            int reg_base = h_idx * 3 * num_deals + deal;
            if (acting == 0) {
                float r0 = (cev0_0 - node_ev0) * regret_scale;
                float r1 = (cev0_1 - node_ev0) * regret_scale;
                float r2 = (cev0_2 - node_ev0) * regret_scale;
                printf("  Regrets (P0): %.4f, %.4f, %.4f (base=%d)\n", r0, r1, r2, reg_base);
            } else {
                float r0 = (cev1_0 - node_ev1) * regret_scale;
                float r1 = (cev1_1 - node_ev1) * regret_scale;
                float r2 = (cev1_2 - node_ev1) * regret_scale;
                printf("  Regrets (P1): %.4f, %.4f, %.4f (base=%d)\n", r0, r1, r2, reg_base);
            }
        }
    }

    // Copy to output
    for (int i = 0; i < num_nodes; i++) {
        debug_ev0[i] = ev0[i];
        debug_ev1[i] = ev1[i];
    }
}
''', 'debug_cfr')


def main():
    game = RiverMicro()

    # Build solver data
    from gpu_poker_cfr.solvers.gpu_river_cfr_v4 import GPURiverCFRv4
    solver = GPURiverCFRv4(game)

    # Initialize strategy to uniform
    solver._compute_strategy()

    # Debug arrays
    debug_ev0 = cp.zeros(solver.num_nodes, dtype=cp.float32)
    debug_ev1 = cp.zeros(solver.num_nodes, dtype=cp.float32)

    print("Running debug kernel...")
    DEBUG_KERNEL(
        (1,), (1,),
        (solver.num_nodes,
         solver.node_type_gpu, solver.node_player_gpu, solver.node_pot_gpu,
         solver.node_num_actions_gpu, solver.node_fold_player_gpu, solver.node_child_idx_gpu,
         solver.num_deals, solver.deal_p0_value, solver.deal_p1_value,
         solver.node_h_p0, solver.node_h_p1,
         solver._strategy, solver._cumulative_regret,
         solver.num_infosets, 1.0 / solver.num_deals,
         debug_ev0, debug_ev1)
    )
    cp.cuda.Stream.null.synchronize()

    print("\nFinal EVs:")
    print("EV0:", cp.asnumpy(debug_ev0))
    print("EV1:", cp.asnumpy(debug_ev1))


if __name__ == "__main__":
    main()
