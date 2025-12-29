"""
GPU CFR v4 - Single CUDA kernel for entire iteration.

One thread per deal, each processes entire tree.
Eliminates all Python loop overhead.
"""

import numpy as np
import cupy as cp
from typing import List, Tuple

from gpu_poker_cfr.games.river_toy import RiverMicro, RiverToyMini, RiverAction
from gpu_poker_cfr.games.hand_eval import evaluate_7cards


# CUDA kernel for CFR iteration - one thread per deal
CFR_KERNEL = cp.RawKernel(r'''
extern "C" __global__ void cfr_iteration(
    // Tree structure (constant)
    const int num_nodes,
    const int* __restrict__ node_type,
    const int* __restrict__ node_player,
    const float* __restrict__ node_pot,
    const int* __restrict__ node_num_actions,
    const int* __restrict__ node_fold_player,
    const int* __restrict__ node_child_idx,  // (num_nodes * 3)

    // Deal info
    const int num_deals,
    const int* __restrict__ deal_p0_value,
    const int* __restrict__ deal_p1_value,

    // Infoset mapping: (num_nodes * num_deals)
    const int* __restrict__ node_h_p0,
    const int* __restrict__ node_h_p1,

    // Strategy: (num_infosets * 3 * num_deals) - row major
    const float* __restrict__ strat,

    // Regrets: (num_infosets * 3 * num_deals)
    float* __restrict__ cumulative_regret,

    const int num_infosets,
    const float regret_scale
) {
    int deal = blockIdx.x * blockDim.x + threadIdx.x;
    if (deal >= num_deals) return;

    // Local EV storage - max 32 nodes should be enough for river
    float ev0[32];
    float ev1[32];

    int p0_val = deal_p0_value[deal];
    int p1_val = deal_p1_value[deal];

    // Process nodes backwards (tree structure ensures children before parents)
    for (int i = num_nodes - 1; i >= 0; i--) {
        int ntype = node_type[i];
        float pot = node_pot[i];
        float half_pot = pot * 0.5f;

        if (ntype == 0) {  // FOLD
            int fp = node_fold_player[i];
            if (fp == 0) {
                ev0[i] = -half_pot;
                ev1[i] = half_pot;
            } else {
                ev0[i] = half_pot;
                ev1[i] = -half_pot;
            }
        }
        else if (ntype == 1) {  // SHOWDOWN
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
        }
        else {  // DECISION (ntype == 2)
            int acting = node_player[i];
            int n_act = node_num_actions[i];

            // Get infoset index
            int h_idx;
            if (acting == 0) {
                h_idx = node_h_p0[i * num_deals + deal];
            } else {
                h_idx = node_h_p1[i * num_deals + deal];
            }

            if (h_idx < 0) continue;

            // Get strategy and child EVs
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
            float cev0_0 = 0.0f, cev0_1 = 0.0f, cev0_2 = 0.0f;
            float cev1_0 = 0.0f, cev1_1 = 0.0f, cev1_2 = 0.0f;

            // Strategy index: h_idx * 3 * num_deals + action * num_deals + deal
            int strat_base = h_idx * 3 * num_deals + deal;

            int c0 = node_child_idx[i * 3 + 0];
            int c1 = node_child_idx[i * 3 + 1];
            int c2 = node_child_idx[i * 3 + 2];

            if (n_act >= 1 && c0 >= 0) {
                s0 = strat[strat_base];
                cev0_0 = ev0[c0];
                cev1_0 = ev1[c0];
            }
            if (n_act >= 2 && c1 >= 0) {
                s1 = strat[strat_base + num_deals];
                cev0_1 = ev0[c1];
                cev1_1 = ev1[c1];
            }
            if (n_act >= 3 && c2 >= 0) {
                s2 = strat[strat_base + 2 * num_deals];
                cev0_2 = ev0[c2];
                cev1_2 = ev1[c2];
            }

            // Weighted EV
            float node_ev0 = s0 * cev0_0 + s1 * cev0_1 + s2 * cev0_2;
            float node_ev1 = s0 * cev1_0 + s1 * cev1_1 + s2 * cev1_2;

            ev0[i] = node_ev0;
            ev1[i] = node_ev1;

            // Regret updates for acting player
            int reg_base = h_idx * 3 * num_deals + deal;

            if (acting == 0) {
                if (n_act >= 1) atomicAdd(&cumulative_regret[reg_base], (cev0_0 - node_ev0) * regret_scale);
                if (n_act >= 2) atomicAdd(&cumulative_regret[reg_base + num_deals], (cev0_1 - node_ev0) * regret_scale);
                if (n_act >= 3) atomicAdd(&cumulative_regret[reg_base + 2 * num_deals], (cev0_2 - node_ev0) * regret_scale);
            } else {
                if (n_act >= 1) atomicAdd(&cumulative_regret[reg_base], (cev1_0 - node_ev1) * regret_scale);
                if (n_act >= 2) atomicAdd(&cumulative_regret[reg_base + num_deals], (cev1_1 - node_ev1) * regret_scale);
                if (n_act >= 3) atomicAdd(&cumulative_regret[reg_base + 2 * num_deals], (cev1_2 - node_ev1) * regret_scale);
            }
        }
    }
}
''', 'cfr_iteration')


# Kernel for computing strategy from regrets
STRATEGY_KERNEL = cp.RawKernel(r'''
extern "C" __global__ void compute_strategy(
    const float* __restrict__ cumulative_regret,
    float* __restrict__ strategy,
    const int* __restrict__ infoset_num_actions,
    const int num_infosets,
    const int num_deals
) {
    int h = blockIdx.x;
    int deal = blockIdx.y * blockDim.x + threadIdx.x;

    if (h >= num_infosets || deal >= num_deals) return;

    int n_act = infoset_num_actions[h];
    int base = h * 3 * num_deals + deal;

    // Get positive regrets
    float r0 = (n_act >= 1) ? fmaxf(cumulative_regret[base], 0.0f) : 0.0f;
    float r1 = (n_act >= 2) ? fmaxf(cumulative_regret[base + num_deals], 0.0f) : 0.0f;
    float r2 = (n_act >= 3) ? fmaxf(cumulative_regret[base + 2 * num_deals], 0.0f) : 0.0f;

    float total = r0 + r1 + r2;

    if (total > 0.0f) {
        strategy[base] = r0 / total;
        strategy[base + num_deals] = r1 / total;
        strategy[base + 2 * num_deals] = r2 / total;
    } else {
        // Uniform over valid actions
        float uniform = 1.0f / n_act;
        strategy[base] = (n_act >= 1) ? uniform : 0.0f;
        strategy[base + num_deals] = (n_act >= 2) ? uniform : 0.0f;
        strategy[base + 2 * num_deals] = (n_act >= 3) ? uniform : 0.0f;
    }
}
''', 'compute_strategy')


class GPURiverCFRv4:
    """GPU CFR with fused CUDA kernel - one thread per deal."""

    def __init__(self, game):
        self.game = game
        self.num_deals = game.num_deals

        # Build tree
        self._build_tree()
        self._setup_gpu_arrays()

        self.iterations = 0

    def _build_tree(self):
        """Build tree structure."""
        self.nodes = []
        self._action_map = {}
        self._build_recursive([], 0, 10)

        self.num_nodes = len(self.nodes)

        # Tree arrays
        self.node_type = np.array([n['type'] for n in self.nodes], dtype=np.int32)
        self.node_player = np.array([n.get('player', 0) for n in self.nodes], dtype=np.int32)
        self.node_pot = np.array([n['pot'] for n in self.nodes], dtype=np.float32)
        self.node_num_actions = np.array([n.get('num_actions', 0) for n in self.nodes], dtype=np.int32)
        self.node_fold_player = np.array([n.get('fold_player', -1) for n in self.nodes], dtype=np.int32)

        # Child node indices
        aid_to_idx = {n['action_id']: i for i, n in enumerate(self.nodes)}
        self.node_child_idx = np.full((self.num_nodes, 3), -1, dtype=np.int32)
        for i, n in enumerate(self.nodes):
            if 'child_ids' in n:
                for j, cid in enumerate(n['child_ids']):
                    if cid >= 0 and cid in aid_to_idx:
                        self.node_child_idx[i, j] = aid_to_idx[cid]

        # Infosets
        self._build_infosets()

    def _build_recursive(self, actions, player, pot, to_call=0, num_bets=0):
        BET_SIZE = 10
        MAX_RAISES = 2
        action_id = self._get_action_id(tuple(actions))

        if actions and actions[-1] == 'F':
            self.nodes.append({
                'type': 0, 'action_id': action_id,
                'fold_player': (len(actions) - 1) % 2, 'pot': pot
            })
            return

        if len(actions) >= 2:
            if (actions[-1] == 'K' and actions[-2] == 'K') or actions[-1] == 'C':
                self.nodes.append({'type': 1, 'action_id': action_id, 'pot': pot})
                return

        if to_call > 0:
            child_actions = ['F', 'C'] + (['R'] if num_bets < MAX_RAISES else [])
        else:
            child_actions = ['K'] + (['B'] if num_bets < MAX_RAISES else [])

        child_ids = [self._get_action_id(tuple(actions + [a])) for a in child_actions]
        while len(child_ids) < 3:
            child_ids.append(-1)

        self.nodes.append({
            'type': 2, 'action_id': action_id, 'player': player,
            'num_actions': len(child_actions), 'pot': pot, 'child_ids': child_ids
        })

        for a in child_actions:
            new_pot, new_to_call, new_bets = pot, 0, num_bets
            if a == 'C':
                new_pot += to_call
            elif a == 'B':
                new_pot, new_to_call, new_bets = pot + BET_SIZE, BET_SIZE, num_bets + 1
            elif a == 'R':
                new_pot, new_to_call, new_bets = pot + to_call + BET_SIZE, BET_SIZE, num_bets + 1
            self._build_recursive(actions + [a], 1 - player, new_pot, new_to_call, new_bets)

    def _get_action_id(self, actions):
        if actions not in self._action_map:
            self._action_map[actions] = len(self._action_map)
        return self._action_map[actions]

    def _build_infosets(self):
        avail = self.game.available_cards
        self.num_cards = len(avail)
        self.card_to_idx = {c: i for i, c in enumerate(avail)}

        self.infoset_key_to_idx = {}
        idx = 0
        for i, n in enumerate(self.nodes):
            if n['type'] == 2:
                for c in range(self.num_cards):
                    key = (c, n['action_id'])
                    if key not in self.infoset_key_to_idx:
                        self.infoset_key_to_idx[key] = idx
                        idx += 1

        self.num_infosets = idx
        self.infoset_num_actions = np.zeros(idx, dtype=np.int32)
        for key, h in self.infoset_key_to_idx.items():
            c, aid = key
            for n in self.nodes:
                if n.get('action_id') == aid and n['type'] == 2:
                    self.infoset_num_actions[h] = n['num_actions']
                    break

    def _setup_gpu_arrays(self):
        board = list(self.game.board)

        # Pre-compute deal info
        p0_cards = np.zeros(self.num_deals, dtype=np.int32)
        p1_cards = np.zeros(self.num_deals, dtype=np.int32)
        p0_values = np.zeros(self.num_deals, dtype=np.int32)
        p1_values = np.zeros(self.num_deals, dtype=np.int32)

        for deal_idx in range(self.num_deals):
            river, p0, p1 = self.game.get_deal(deal_idx)
            br = board + [river]
            h0 = list(p0) + br
            h1 = list(p1) + br
            while len(h0) < 7: h0.append(h0[-1])
            while len(h1) < 7: h1.append(h1[-1])

            p0_cards[deal_idx] = self.card_to_idx.get(p0[0], 0)
            p1_cards[deal_idx] = self.card_to_idx.get(p1[0], 0)
            p0_values[deal_idx] = evaluate_7cards(np.array(h0, dtype=np.int32))
            p1_values[deal_idx] = evaluate_7cards(np.array(h1, dtype=np.int32))

        # GPU arrays
        self.deal_p0_value = cp.asarray(p0_values)
        self.deal_p1_value = cp.asarray(p1_values)

        # Node infoset indices per deal: (num_nodes, num_deals) - flattened for kernel
        node_h_p0 = np.full((self.num_nodes, self.num_deals), -1, dtype=np.int32)
        node_h_p1 = np.full((self.num_nodes, self.num_deals), -1, dtype=np.int32)

        for i, n in enumerate(self.nodes):
            if n['type'] == 2:
                aid = n['action_id']
                for d in range(self.num_deals):
                    k0 = (int(p0_cards[d]), aid)
                    k1 = (int(p1_cards[d]), aid)
                    if k0 in self.infoset_key_to_idx:
                        node_h_p0[i, d] = self.infoset_key_to_idx[k0]
                    if k1 in self.infoset_key_to_idx:
                        node_h_p1[i, d] = self.infoset_key_to_idx[k1]

        self.node_h_p0 = cp.asarray(node_h_p0)
        self.node_h_p1 = cp.asarray(node_h_p1)

        # Tree arrays on GPU
        self.node_type_gpu = cp.asarray(self.node_type)
        self.node_player_gpu = cp.asarray(self.node_player)
        self.node_pot_gpu = cp.asarray(self.node_pot)
        self.node_num_actions_gpu = cp.asarray(self.node_num_actions)
        self.node_fold_player_gpu = cp.asarray(self.node_fold_player)
        self.node_child_idx_gpu = cp.asarray(self.node_child_idx.flatten())  # Flattened

        # Regrets: (num_infosets, 3, num_deals) - but stored as flat for kernel
        self._cumulative_regret = cp.zeros((self.num_infosets * 3 * self.num_deals,), dtype=cp.float32)
        self._cumulative_strategy = cp.zeros((self.num_infosets * 3 * self.num_deals,), dtype=cp.float32)
        self._strategy = cp.zeros((self.num_infosets * 3 * self.num_deals,), dtype=cp.float32)

        # Initialize uniform strategy
        self.infoset_num_actions_gpu = cp.asarray(self.infoset_num_actions)

    def _compute_strategy(self):
        """Compute strategy using CUDA kernel."""
        block_size = 256
        grid_y = (self.num_deals + block_size - 1) // block_size

        STRATEGY_KERNEL(
            (self.num_infosets, grid_y), (block_size,),
            (self._cumulative_regret, self._strategy, self.infoset_num_actions_gpu,
             np.int32(self.num_infosets), np.int32(self.num_deals))
        )
        return self._strategy

    def iterate(self, n: int = 1):
        for _ in range(n):
            self._single_iteration()
            self.iterations += 1

    def _single_iteration(self):
        """Single iteration using fused CUDA kernel."""
        t = self.iterations + 1
        strat = self._compute_strategy()

        # Launch CFR kernel - one thread per deal
        block_size = 256
        grid_size = (self.num_deals + block_size - 1) // block_size
        # Must use numpy types for scalar arguments in CuPy RawKernel
        regret_scale = np.float32(1.0 / self.num_deals)

        CFR_KERNEL(
            (grid_size,), (block_size,),
            (np.int32(self.num_nodes),
             self.node_type_gpu, self.node_player_gpu, self.node_pot_gpu,
             self.node_num_actions_gpu, self.node_fold_player_gpu, self.node_child_idx_gpu,
             np.int32(self.num_deals), self.deal_p0_value, self.deal_p1_value,
             self.node_h_p0, self.node_h_p1,
             strat, self._cumulative_regret,
             np.int32(self.num_infosets), regret_scale)
        )

        # Update cumulative strategy
        self._cumulative_strategy += t * strat

        # Sync to ensure completion
        cp.cuda.Stream.null.synchronize()

    def solve(self, iterations: int = 1000):
        self.iterate(iterations)

    @property
    def average_strategy(self):
        """Return average strategy reshaped to (num_infosets, 3, num_deals)."""
        strat = self._cumulative_strategy.reshape((self.num_infosets, 3, self.num_deals))
        total = strat.sum(axis=1, keepdims=True)
        safe = cp.where(total > 0, total, 1.0)
        return cp.where(total > 0, strat / safe, 1.0/3)
