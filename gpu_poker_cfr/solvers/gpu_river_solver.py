"""
GPU CFR Solver for Enhanced River Game.

Handles:
- Multiple bet sizes
- Variable pot sizes
- Dynamic game trees
"""

import numpy as np
import cupy as cp
from typing import List, Tuple, Dict

from gpu_poker_cfr.games.hand_eval import evaluate_7cards
from gpu_poker_cfr.games.river_game import RiverGame, build_river_tree


# GPU kernel for strategy computation (regret matching)
STRATEGY_KERNEL = cp.RawKernel(r'''
extern "C" __global__ void compute_strategy(
    const float* __restrict__ cumulative_regret,
    float* __restrict__ strategy,
    const int* __restrict__ infoset_num_actions,
    const int num_infosets,
    const int max_actions
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_infosets) return;

    int n_act = infoset_num_actions[h];
    int base = h * max_actions;

    // Compute positive regrets and sum
    float total = 0.0f;
    for (int a = 0; a < n_act; a++) {
        float r = fmaxf(cumulative_regret[base + a], 0.0f);
        total += r;
    }

    // Regret matching
    if (total > 0.0f) {
        for (int a = 0; a < n_act; a++) {
            strategy[base + a] = fmaxf(cumulative_regret[base + a], 0.0f) / total;
        }
    } else {
        float uniform = 1.0f / n_act;
        for (int a = 0; a < n_act; a++) {
            strategy[base + a] = uniform;
        }
    }

    // Zero out unused actions
    for (int a = n_act; a < max_actions; a++) {
        strategy[base + a] = 0.0f;
    }
}
''', 'compute_strategy')


# GPU kernel for CFR iteration
CFR_KERNEL = cp.RawKernel(r'''
extern "C" __global__ void cfr_iteration(
    // Tree structure
    const int num_nodes,
    const int max_actions,
    const int* __restrict__ node_type,        // 0=FOLD, 1=SHOWDOWN, 2=DECISION
    const int* __restrict__ node_player,      // 0=OOP, 1=IP
    const int* __restrict__ node_num_actions,
    const int* __restrict__ node_fold_player,
    const int* __restrict__ node_children,    // [num_nodes * max_actions]
    const float* __restrict__ node_invested_oop,
    const float* __restrict__ node_invested_ip,

    // Deal info
    const int num_deals,
    const int* __restrict__ deal_p0_value,
    const int* __restrict__ deal_p1_value,

    // Infoset mapping: node_h_p0[node * num_deals + deal] = infoset_idx
    const int* __restrict__ node_h_p0,
    const int* __restrict__ node_h_p1,

    // Strategy and accumulators
    const float* __restrict__ strategy,
    float* __restrict__ cumulative_regret,
    float* __restrict__ cumulative_strategy,

    const int num_infosets,
    const float iteration_weight
) {
    int deal = blockIdx.x * blockDim.x + threadIdx.x;
    if (deal >= num_deals) return;

    // Local arrays for this deal
    float ev0[64];  // EV for player 0 at each node
    float ev1[64];  // EV for player 1 at each node
    float reach0[64];  // Player 0 reach probability
    float reach1[64];  // Player 1 reach probability

    int p0_val = deal_p0_value[deal];
    int p1_val = deal_p1_value[deal];

    // Initialize root reach
    reach0[0] = 1.0f;
    reach1[0] = 1.0f;

    // Forward pass: compute reach probabilities and accumulate strategy
    for (int i = 0; i < num_nodes; i++) {
        int ntype = node_type[i];
        if (ntype != 2) continue;  // Only decision nodes

        int player = node_player[i];
        int n_act = node_num_actions[i];

        // Get infoset index for this (node, deal)
        int h_idx;
        if (player == 0) {
            h_idx = node_h_p0[i * num_deals + deal];
        } else {
            h_idx = node_h_p1[i * num_deals + deal];
        }

        if (h_idx < 0) {
            // Invalid deal for this node - zero out child reaches
            for (int a = 0; a < n_act; a++) {
                int child = node_children[i * max_actions + a];
                if (child >= 0) {
                    reach0[child] = 0.0f;
                    reach1[child] = 0.0f;
                }
            }
            continue;
        }

        // Get strategy for this infoset
        int strat_base = h_idx * max_actions;
        float own_reach = (player == 0) ? reach0[i] : reach1[i];

        // Accumulate strategy and propagate reach
        for (int a = 0; a < n_act; a++) {
            float s = strategy[strat_base + a];
            int child = node_children[i * max_actions + a];

            // Accumulate strategy weighted by own reach
            atomicAdd(&cumulative_strategy[strat_base + a], s * own_reach * iteration_weight);

            // Propagate reach to child
            if (child >= 0) {
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

    // Backward pass: compute EVs and update regrets
    for (int i = num_nodes - 1; i >= 0; i--) {
        int ntype = node_type[i];

        if (ntype == 0) {  // FOLD
            int fp = node_fold_player[i];
            float inv_oop = node_invested_oop[i];
            float inv_ip = node_invested_ip[i];
            if (fp == 0) {  // OOP folds
                ev0[i] = -inv_oop;
                ev1[i] = inv_oop;
            } else {  // IP folds
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
        else {  // DECISION
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

            // Compute node EV as weighted sum of child EVs
            float node_ev0 = 0.0f;
            float node_ev1 = 0.0f;
            float child_ev0[8];  // Max 8 actions
            float child_ev1[8];

            for (int a = 0; a < n_act; a++) {
                int child = node_children[i * max_actions + a];
                float s = strategy[strat_base + a];
                if (child >= 0) {
                    child_ev0[a] = ev0[child];
                    child_ev1[a] = ev1[child];
                } else {
                    child_ev0[a] = 0.0f;
                    child_ev1[a] = 0.0f;
                }
                node_ev0 += s * child_ev0[a];
                node_ev1 += s * child_ev1[a];
            }

            ev0[i] = node_ev0;
            ev1[i] = node_ev1;

            // Update regrets weighted by opponent reach
            float opp_reach = (player == 0) ? reach1[i] : reach0[i];

            for (int a = 0; a < n_act; a++) {
                float regret;
                if (player == 0) {
                    regret = (child_ev0[a] - node_ev0) * opp_reach;
                } else {
                    regret = (child_ev1[a] - node_ev1) * opp_reach;
                }
                atomicAdd(&cumulative_regret[strat_base + a], regret);
            }
        }
    }
}
''', 'cfr_iteration')


class GPURiverSolver:
    """GPU CFR solver for river games with multiple bet sizes."""

    def __init__(self, game: RiverGame):
        self.game = game
        self.num_deals = game.num_deals

        # Build game tree
        self.nodes, self.node_children = build_river_tree(game)
        self.num_nodes = len(self.nodes)
        self.max_actions = self.node_children.shape[1]

        self._build_tree_arrays()
        self._build_infosets()
        self._setup_gpu_arrays()
        self.iterations = 0

    def _build_tree_arrays(self):
        """Convert tree to numpy arrays for GPU."""
        self.node_type = np.array([n['type'] for n in self.nodes], dtype=np.int32)
        self.node_player = np.array([n.get('player', -1) for n in self.nodes], dtype=np.int32)
        self.node_pot = np.array([n['pot'] for n in self.nodes], dtype=np.float32)
        self.node_num_actions = np.array(
            [len(n.get('actions', [])) for n in self.nodes], dtype=np.int32
        )
        self.node_fold_player = np.array(
            [n.get('fold_player', -1) for n in self.nodes], dtype=np.int32
        )

        # For fold nodes, compute the EV
        # invested_oop, invested_ip stored for EV calculation
        self.node_invested_oop = np.array(
            [n.get('invested_oop', 0) for n in self.nodes], dtype=np.float32
        )
        self.node_invested_ip = np.array(
            [n.get('invested_ip', 0) for n in self.nodes], dtype=np.float32
        )

    def _build_infosets(self):
        """Build infoset mapping."""
        # Normalize hands
        self.oop_hands = list(set(tuple(sorted(h, reverse=True)) for h in self.game.oop_range))
        self.ip_hands = list(set(tuple(sorted(h, reverse=True)) for h in self.game.ip_range))

        self.oop_hand_to_idx = {h: i for i, h in enumerate(self.oop_hands)}
        self.ip_hand_to_idx = {h: i for i, h in enumerate(self.ip_hands)}

        # Map (player, hand_idx, node_idx) -> infoset_idx
        # Each decision node creates infosets for the acting player's hands
        self.infoset_key_to_idx = {}
        self.infoset_to_node = {}  # Map infoset back to node for action names
        idx = 0

        for node_idx, node in enumerate(self.nodes):
            if node['type'] == 2:  # Decision node
                player = node['player']
                hands = self.oop_hands if player == 0 else self.ip_hands

                for hand_idx in range(len(hands)):
                    key = (player, hand_idx, node_idx)
                    if key not in self.infoset_key_to_idx:
                        self.infoset_key_to_idx[key] = idx
                        self.infoset_to_node[idx] = node_idx
                        idx += 1

        self.num_infosets = idx

        # Store number of actions per infoset
        self.infoset_num_actions = np.zeros(self.num_infosets, dtype=np.int32)
        for key, h_idx in self.infoset_key_to_idx.items():
            player, hand_idx, node_idx = key
            self.infoset_num_actions[h_idx] = self.node_num_actions[node_idx]

    def _setup_gpu_arrays(self):
        """Setup GPU arrays."""
        board = list(self.game.board)

        # Compute hand values
        p0_values = np.zeros(self.num_deals, dtype=np.int32)
        p1_values = np.zeros(self.num_deals, dtype=np.int32)

        for deal_idx in range(self.num_deals):
            oop_hand, ip_hand = self.game.get_deal(deal_idx)
            h0 = list(oop_hand) + board
            h1 = list(ip_hand) + board
            p0_values[deal_idx] = evaluate_7cards(np.array(h0, dtype=np.int32))
            p1_values[deal_idx] = evaluate_7cards(np.array(h1, dtype=np.int32))

        self.deal_p0_value = cp.asarray(p0_values)
        self.deal_p1_value = cp.asarray(p1_values)

        # Map (node, deal) -> infoset index
        node_h_p0 = np.full((self.num_nodes, self.num_deals), -1, dtype=np.int32)
        node_h_p1 = np.full((self.num_nodes, self.num_deals), -1, dtype=np.int32)

        for deal_idx in range(self.num_deals):
            oop_hand, ip_hand = self.game.get_deal(deal_idx)
            oop_hand = tuple(sorted(oop_hand, reverse=True))
            ip_hand = tuple(sorted(ip_hand, reverse=True))

            oop_hand_idx = self.oop_hand_to_idx.get(oop_hand, -1)
            ip_hand_idx = self.ip_hand_to_idx.get(ip_hand, -1)

            for node_idx, node in enumerate(self.nodes):
                if node['type'] == 2:
                    player = node['player']
                    if player == 0 and oop_hand_idx >= 0:
                        key = (0, oop_hand_idx, node_idx)
                        if key in self.infoset_key_to_idx:
                            node_h_p0[node_idx, deal_idx] = self.infoset_key_to_idx[key]
                    elif player == 1 and ip_hand_idx >= 0:
                        key = (1, ip_hand_idx, node_idx)
                        if key in self.infoset_key_to_idx:
                            node_h_p1[node_idx, deal_idx] = self.infoset_key_to_idx[key]

        self.node_h_p0 = cp.asarray(node_h_p0)
        self.node_h_p1 = cp.asarray(node_h_p1)

        # Tree arrays on GPU
        self.node_type_gpu = cp.asarray(self.node_type)
        self.node_player_gpu = cp.asarray(self.node_player)
        self.node_pot_gpu = cp.asarray(self.node_pot)
        self.node_num_actions_gpu = cp.asarray(self.node_num_actions)
        self.node_fold_player_gpu = cp.asarray(self.node_fold_player)
        self.node_children_gpu = cp.asarray(self.node_children.flatten())
        self.node_invested_oop_gpu = cp.asarray(self.node_invested_oop)
        self.node_invested_ip_gpu = cp.asarray(self.node_invested_ip)

        # Regret and strategy arrays - per infoset
        self._cumulative_regret = cp.zeros((self.num_infosets, self.max_actions), dtype=cp.float32)
        self._cumulative_strategy = cp.zeros((self.num_infosets, self.max_actions), dtype=cp.float32)
        self._strategy = cp.zeros((self.num_infosets, self.max_actions), dtype=cp.float32)

        self.infoset_num_actions_gpu = cp.asarray(self.infoset_num_actions)

    def _compute_strategy_gpu(self):
        """Compute strategy from regrets on GPU."""
        block_size = 256
        grid_size = (self.num_infosets + block_size - 1) // block_size

        STRATEGY_KERNEL(
            (grid_size,), (block_size,),
            (self._cumulative_regret, self._strategy, self.infoset_num_actions_gpu,
             np.int32(self.num_infosets), np.int32(self.max_actions))
        )
        return self._strategy

    def _compute_strategy_cpu(self):
        """Compute strategy from regrets on CPU (simpler for debugging)."""
        regrets = cp.asnumpy(self._cumulative_regret)
        strategy = np.zeros_like(regrets)

        for h in range(self.num_infosets):
            n_act = self.infoset_num_actions[h]
            pos_regrets = np.maximum(regrets[h, :n_act], 0)
            total = pos_regrets.sum()

            if total > 0:
                strategy[h, :n_act] = pos_regrets / total
            else:
                strategy[h, :n_act] = 1.0 / n_act

        self._strategy = cp.asarray(strategy)
        return self._strategy

    def _cfr_iteration_cpu(self):
        """Run one CFR iteration on CPU (for correctness verification)."""
        t = self.iterations + 1
        strat = cp.asnumpy(self._compute_strategy_cpu())

        p0_values = cp.asnumpy(self.deal_p0_value)
        p1_values = cp.asnumpy(self.deal_p1_value)
        node_h_p0 = cp.asnumpy(self.node_h_p0)
        node_h_p1 = cp.asnumpy(self.node_h_p1)

        cum_regret = cp.asnumpy(self._cumulative_regret)
        cum_strat = cp.asnumpy(self._cumulative_strategy)

        for deal in range(self.num_deals):
            # EVs for each node (indexed by node_idx)
            ev0 = np.zeros(self.num_nodes)
            ev1 = np.zeros(self.num_nodes)
            reach0 = np.zeros(self.num_nodes)  # Player 0 reach
            reach1 = np.zeros(self.num_nodes)  # Player 1 reach
            reach0[0] = 1.0
            reach1[0] = 1.0

            p0_val = p0_values[deal]
            p1_val = p1_values[deal]

            # Forward pass: compute reach probabilities
            for i in range(self.num_nodes):
                node = self.nodes[i]
                if node['type'] != 2:
                    continue

                player = node['player']
                n_act = len(node['actions'])

                h_idx = node_h_p0[i, deal] if player == 0 else node_h_p1[i, deal]
                if h_idx < 0:
                    continue

                # Get strategy for this infoset
                s = strat[h_idx, :n_act]

                # Accumulate strategy weighted by own reach
                own_reach = reach0[i] if player == 0 else reach1[i]
                cum_strat[h_idx, :n_act] += s * own_reach * t

                # Propagate reach to children
                for a in range(n_act):
                    child = self.node_children[i, a]
                    if child >= 0:
                        if player == 0:
                            reach0[child] = reach0[i] * s[a]
                            reach1[child] = reach1[i]
                        else:
                            reach0[child] = reach0[i]
                            reach1[child] = reach1[i] * s[a]

            # Backward pass: compute EVs and regrets
            for i in range(self.num_nodes - 1, -1, -1):
                node = self.nodes[i]
                ntype = node['type']

                if ntype == 0:  # FOLD
                    # Folder loses their investment, opponent wins it
                    fp = node['fold_player']
                    inv_oop = node['invested_oop']
                    inv_ip = node['invested_ip']
                    if fp == 0:  # OOP folds
                        ev0[i] = -inv_oop
                        ev1[i] = inv_oop
                    else:  # IP folds
                        ev0[i] = inv_ip
                        ev1[i] = -inv_ip

                elif ntype == 1:  # SHOWDOWN
                    inv_oop = node['invested_oop']
                    inv_ip = node['invested_ip']
                    if p0_val > p1_val:
                        ev0[i] = inv_ip
                        ev1[i] = -inv_ip
                    elif p1_val > p0_val:
                        ev0[i] = -inv_oop
                        ev1[i] = inv_oop
                    else:
                        ev0[i] = 0
                        ev1[i] = 0

                else:  # DECISION
                    player = node['player']
                    n_act = len(node['actions'])

                    h_idx = node_h_p0[i, deal] if player == 0 else node_h_p1[i, deal]
                    if h_idx < 0:
                        continue

                    s = strat[h_idx, :n_act]

                    # Compute node EV as weighted sum of child EVs
                    child_ev0 = np.zeros(n_act)
                    child_ev1 = np.zeros(n_act)
                    for a in range(n_act):
                        child = self.node_children[i, a]
                        if child >= 0:
                            child_ev0[a] = ev0[child]
                            child_ev1[a] = ev1[child]

                    node_ev0 = np.dot(s, child_ev0)
                    node_ev1 = np.dot(s, child_ev1)
                    ev0[i] = node_ev0
                    ev1[i] = node_ev1

                    # Update regrets weighted by opponent reach
                    opp_reach = reach1[i] if player == 0 else reach0[i]
                    if player == 0:
                        regrets = (child_ev0 - node_ev0) * opp_reach
                    else:
                        regrets = (child_ev1 - node_ev1) * opp_reach

                    cum_regret[h_idx, :n_act] += regrets

        # CFR+ floor
        cum_regret = np.maximum(cum_regret, 0)

        self._cumulative_regret = cp.asarray(cum_regret)
        self._cumulative_strategy = cp.asarray(cum_strat)

    def _cfr_iteration_gpu(self):
        """Run one CFR iteration on GPU."""
        t = self.iterations + 1
        strat = self._compute_strategy_gpu()

        block_size = 256
        grid_size = (self.num_deals + block_size - 1) // block_size
        iteration_weight = np.float32(t)

        CFR_KERNEL(
            (grid_size,), (block_size,),
            (np.int32(self.num_nodes),
             np.int32(self.max_actions),
             self.node_type_gpu,
             self.node_player_gpu,
             self.node_num_actions_gpu,
             self.node_fold_player_gpu,
             self.node_children_gpu,
             self.node_invested_oop_gpu,
             self.node_invested_ip_gpu,
             np.int32(self.num_deals),
             self.deal_p0_value,
             self.deal_p1_value,
             self.node_h_p0,
             self.node_h_p1,
             strat,
             self._cumulative_regret,
             self._cumulative_strategy,
             np.int32(self.num_infosets),
             iteration_weight)
        )

        # CFR+ floor
        self._cumulative_regret = cp.maximum(self._cumulative_regret, 0)
        cp.cuda.Stream.null.synchronize()

    def iterate(self, n: int = 1, use_gpu: bool = True):
        for _ in range(n):
            if use_gpu:
                self._cfr_iteration_gpu()
            else:
                self._cfr_iteration_cpu()
            self.iterations += 1

    def solve(self, iterations: int = 1000, use_gpu: bool = True):
        self.iterate(iterations, use_gpu=use_gpu)

    def get_strategy_for_hand(self, player: int, hand: Tuple[int, int], node_idx: int) -> Dict[str, float]:
        """Get average strategy for a hand at a specific node."""
        hand = tuple(sorted(hand, reverse=True))

        if player == 0:
            hand_idx = self.oop_hand_to_idx.get(hand, -1)
        else:
            hand_idx = self.ip_hand_to_idx.get(hand, -1)

        if hand_idx < 0:
            return {}

        key = (player, hand_idx, node_idx)
        if key not in self.infoset_key_to_idx:
            return {}

        h = self.infoset_key_to_idx[key]
        node = self.nodes[node_idx]
        actions = node.get('actions', [])
        n_act = len(actions)

        strat_sum = cp.asnumpy(self._cumulative_strategy[h, :n_act])
        total = strat_sum.sum()

        if total > 0:
            probs = strat_sum / total
        else:
            probs = np.ones(n_act) / n_act

        return {actions[i]: float(probs[i]) for i in range(n_act)}

    def get_aggregate_strategy(self, node_idx: int) -> Dict[str, float]:
        """Get deal-weighted aggregate strategy at a node."""
        node = self.nodes[node_idx]
        if node['type'] != 2:
            return {}

        player = node['player']
        hands = self.oop_hands if player == 0 else self.ip_hands
        actions = node.get('actions', [])

        # Count deals per hand
        hand_deals = {}
        for d in range(self.num_deals):
            oop_hand, ip_hand = self.game.get_deal(d)
            hand = tuple(sorted(oop_hand if player == 0 else ip_hand, reverse=True))
            hand_deals[hand] = hand_deals.get(hand, 0) + 1

        totals = {a: 0.0 for a in actions}
        total_deals = 0

        for hand in hands:
            deals = hand_deals.get(hand, 0)
            if deals == 0:
                continue

            strat = self.get_strategy_for_hand(player, hand, node_idx)
            for action, prob in strat.items():
                totals[action] += prob * deals
            total_deals += deals

        if total_deals > 0:
            return {a: v / total_deals for a, v in totals.items()}
        return {}

    def print_strategy(self):
        """Print strategies at all decision nodes."""
        from gpu_poker_cfr.games.cards import card_name

        for node_idx, node in enumerate(self.nodes):
            if node['type'] != 2:
                continue

            player = 'OOP' if node['player'] == 0 else 'IP'
            actions = node['actions']
            print(f"\n=== Node {node_idx}: {player} (pot={node['pot']:.0f}) ===")
            print(f"Actions: {actions}")

            agg = self.get_aggregate_strategy(node_idx)
            print(f"Aggregate: {', '.join(f'{a}={v*100:.1f}%' for a, v in agg.items())}")
