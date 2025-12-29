"""
GPU CFR Solver for Turn Game with Chance Nodes.

Key differences from river solver:
- Handles NODE_CHANCE (type=3) for river card dealing
- Precomputes hand values for all (deal, river_card) combinations
- At chance nodes: averages EVs over valid river outcomes
- Separate processing for turn nodes (before chance) and river nodes (after chance)
"""

import numpy as np
import cupy as cp
from typing import List, Tuple, Dict

from gpu_poker_cfr.games.hand_eval import evaluate_7cards
from gpu_poker_cfr.games.turn_toy_game import (
    TurnGame, build_turn_tree,
    NODE_FOLD, NODE_SHOWDOWN, NODE_DECISION, NODE_CHANCE
)


# GPU kernel for strategy computation (same as river)
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


# GPU kernel for turn CFR with chance nodes
# This kernel processes one deal at a time, looping over valid rivers internally
# Uses unified 64-element arrays for all nodes to avoid indexing issues
TURN_CFR_KERNEL = cp.RawKernel(r'''
extern "C" __global__ void turn_cfr_iteration(
    // Tree structure
    const int num_nodes,
    const int max_actions,
    const int* __restrict__ node_type,        // 0=FOLD, 1=SHOWDOWN, 2=DECISION, 3=CHANCE
    const int* __restrict__ node_player,
    const int* __restrict__ node_num_actions,
    const int* __restrict__ node_fold_player,
    const int* __restrict__ node_children,    // [num_nodes * max_actions]
    const float* __restrict__ node_invested_oop,
    const float* __restrict__ node_invested_ip,
    const int* __restrict__ node_is_river,    // 1 if node is in river subtree, 0 if turn

    // Deal info
    const int num_deals,
    const int num_possible_rivers,            // 48 for a 4-card board

    // River validity and hand values per (deal, river)
    const int* __restrict__ deal_river_valid,     // [num_deals * 48] 1 if valid
    const int* __restrict__ deal_river_p0_value,  // [num_deals * 48] 7-card hand value
    const int* __restrict__ deal_river_p1_value,

    // Infoset mapping: node_h_p*[node * num_deals + deal] = infoset_idx
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

    // Count valid rivers for this deal
    int num_valid_rivers = 0;
    for (int r = 0; r < num_possible_rivers; r++) {
        if (deal_river_valid[deal * num_possible_rivers + r]) {
            num_valid_rivers++;
        }
    }
    if (num_valid_rivers == 0) return;

    float river_weight = 1.0f / num_valid_rivers;

    // Unified arrays for all nodes (64 should be enough for turn toy game)
    float ev0[64];
    float ev1[64];
    float reach0[64];
    float reach1[64];

    // Initialize all to zero
    for (int i = 0; i < 64; i++) {
        ev0[i] = 0.0f;
        ev1[i] = 0.0f;
        reach0[i] = 0.0f;
        reach1[i] = 0.0f;
    }

    // Root reach = 1
    reach0[0] = 1.0f;
    reach1[0] = 1.0f;

    // ========================================
    // FORWARD PASS - All nodes (turn then river)
    // ========================================
    for (int i = 0; i < num_nodes; i++) {
        int ntype = node_type[i];

        if (ntype == 3) {  // CHANCE
            // Chance node - just propagate reach to child
            int child = node_children[i * max_actions + 0];
            if (child >= 0 && child < 64) {
                reach0[child] = reach0[i];
                reach1[child] = reach1[i];
            }
            continue;
        }

        if (ntype != 2) continue;  // Not DECISION

        int player = node_player[i];
        int n_act = node_num_actions[i];

        int h_idx;
        if (player == 0) {
            h_idx = node_h_p0[i * num_deals + deal];
        } else {
            h_idx = node_h_p1[i * num_deals + deal];
        }

        if (h_idx < 0) {
            // Invalid - zero child reaches
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
        float own_reach = (player == 0) ? reach0[i] : reach1[i];

        // Accumulate strategy - NO river_weight here
        // River strategy accumulation happens in backward pass (once per river)
        if (!node_is_river[i]) {
            for (int a = 0; a < n_act; a++) {
                float s = strategy[strat_base + a];
                atomicAdd(&cumulative_strategy[strat_base + a],
                          s * own_reach * iteration_weight);
            }
        }

        // Propagate reach to children
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

    // ========================================
    // BACKWARD PASS - For each valid river
    // ========================================
    // We need to accumulate chance node EVs across all rivers
    float chance_ev0[8];
    float chance_ev1[8];
    int chance_node_ids[8];
    int num_chance_nodes = 0;

    for (int i = 0; i < num_nodes && num_chance_nodes < 8; i++) {
        if (node_type[i] == 3) {  // CHANCE
            chance_node_ids[num_chance_nodes] = i;
            chance_ev0[num_chance_nodes] = 0.0f;
            chance_ev1[num_chance_nodes] = 0.0f;
            num_chance_nodes++;
        }
    }

    // For each valid river
    for (int r = 0; r < num_possible_rivers; r++) {
        if (!deal_river_valid[deal * num_possible_rivers + r]) continue;

        int p0_val = deal_river_p0_value[deal * num_possible_rivers + r];
        int p1_val = deal_river_p1_value[deal * num_possible_rivers + r];

        // Backward pass for river nodes with this river's hand values
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
                float own_reach = (player == 0) ? reach0[i] : reach1[i];

                // Accumulate strategy for river nodes (once per river card)
                // NOTE: Do NOT multiply by river_weight - strategy accumulates
                // across all rivers, same as regrets.
                for (int a = 0; a < n_act; a++) {
                    float s = strategy[strat_base + a];
                    atomicAdd(&cumulative_strategy[strat_base + a],
                              s * own_reach * iteration_weight);
                }

                // Compute node EV
                float node_ev0 = 0.0f;
                float node_ev1 = 0.0f;
                float child_ev0[4];
                float child_ev1[4];

                for (int a = 0; a < n_act; a++) {
                    int child = node_children[i * max_actions + a];
                    float s = strategy[strat_base + a];
                    if (child >= 0 && child < 64) {
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

                // Update regrets weighted by opponent reach only
                // NOTE: Do NOT multiply by river_weight here - regrets accumulate
                // across all rivers, not averaged. The averaging happens at chance
                // nodes for EVs, but regrets should sum.
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

        // Accumulate chance node EVs
        for (int c = 0; c < num_chance_nodes; c++) {
            int chance_idx = chance_node_ids[c];
            int child = node_children[chance_idx * max_actions + 0];
            if (child >= 0 && child < 64) {
                chance_ev0[c] += ev0[child] * river_weight;
                chance_ev1[c] += ev1[child] * river_weight;
            }
        }
    }

    // ========================================
    // BACKWARD PASS - Turn portion
    // ========================================
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
            // Use accumulated EV from all rivers
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

            // Compute node EV
            float node_ev0 = 0.0f;
            float node_ev1 = 0.0f;
            float child_ev0[4];
            float child_ev1[4];

            for (int a = 0; a < n_act; a++) {
                int child = node_children[i * max_actions + a];
                float s = strategy[strat_base + a];
                if (child >= 0 && child < 64) {
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

            // Update regrets
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
''', 'turn_cfr_iteration')


class GPUTurnSolver:
    """GPU CFR solver for turn games with chance nodes."""

    def __init__(self, game: TurnGame):
        self.game = game
        self.num_deals = game.num_deals

        # Build game tree
        self.nodes, self.node_children = build_turn_tree(game)
        self.num_nodes = len(self.nodes)
        self.max_actions = self.node_children.shape[1]

        print(f"Turn solver: {self.num_nodes} nodes, {self.num_deals} deals")

        self._build_tree_arrays()
        self._build_infosets()
        self._precompute_river_hand_values()
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
        self.node_invested_oop = np.array(
            [n.get('invested_oop', 0) for n in self.nodes], dtype=np.float32
        )
        self.node_invested_ip = np.array(
            [n.get('invested_ip', 0) for n in self.nodes], dtype=np.float32
        )

        # Mark which nodes are in river subtree (after chance nodes)
        self.node_is_river = np.zeros(self.num_nodes, dtype=np.int32)
        self._mark_river_nodes()

    def _mark_river_nodes(self):
        """Mark nodes that are in the river subtree (after chance nodes)."""
        # BFS from chance node children
        from collections import deque
        queue = deque()

        for i, node in enumerate(self.nodes):
            if node['type'] == NODE_CHANCE:
                for child in self.node_children[i]:
                    if child >= 0:
                        queue.append(child)

        while queue:
            idx = queue.popleft()
            if self.node_is_river[idx]:
                continue
            self.node_is_river[idx] = 1

            for child in self.node_children[idx]:
                if child >= 0:
                    queue.append(child)

        river_count = self.node_is_river.sum()
        print(f"Turn nodes: {self.num_nodes - river_count}, River nodes: {river_count}")

    def _build_infosets(self):
        """Build infoset mapping."""
        # Normalize hands
        self.oop_hands = list(set(tuple(sorted(h, reverse=True)) for h in self.game.oop_range))
        self.ip_hands = list(set(tuple(sorted(h, reverse=True)) for h in self.game.ip_range))

        self.oop_hand_to_idx = {h: i for i, h in enumerate(self.oop_hands)}
        self.ip_hand_to_idx = {h: i for i, h in enumerate(self.ip_hands)}

        # Map (player, hand_idx, node_idx) -> infoset_idx
        self.infoset_key_to_idx = {}
        self.infoset_to_node = {}
        idx = 0

        for node_idx, node in enumerate(self.nodes):
            if node['type'] == NODE_DECISION:
                player = node['player']
                hands = self.oop_hands if player == 0 else self.ip_hands

                for hand_idx in range(len(hands)):
                    key = (player, hand_idx, node_idx)
                    if key not in self.infoset_key_to_idx:
                        self.infoset_key_to_idx[key] = idx
                        self.infoset_to_node[idx] = node_idx
                        idx += 1

        self.num_infosets = idx
        print(f"Turn solver: {self.num_infosets} infosets")

        # Store number of actions per infoset
        self.infoset_num_actions = np.zeros(self.num_infosets, dtype=np.int32)
        for key, h_idx in self.infoset_key_to_idx.items():
            player, hand_idx, node_idx = key
            self.infoset_num_actions[h_idx] = self.node_num_actions[node_idx]

    def _precompute_river_hand_values(self):
        """Precompute 7-card hand values for all (deal, river) combinations."""
        board = list(self.game.board)  # 4 cards
        possible_rivers = self.game.possible_rivers  # 48 cards
        num_rivers = len(possible_rivers)

        # Create mapping from card to river index
        self.river_to_idx = {c: i for i, c in enumerate(possible_rivers)}

        # For each (deal, river), compute hand values and validity
        self.deal_river_valid = np.zeros((self.num_deals, num_rivers), dtype=np.int32)
        self.deal_river_p0_value = np.zeros((self.num_deals, num_rivers), dtype=np.int32)
        self.deal_river_p1_value = np.zeros((self.num_deals, num_rivers), dtype=np.int32)

        for deal_idx in range(self.num_deals):
            oop_hand, ip_hand = self.game.get_deal(deal_idx)
            blocked = set(oop_hand) | set(ip_hand)

            for river_idx, river_card in enumerate(possible_rivers):
                if river_card in blocked:
                    # Invalid river for this deal
                    self.deal_river_valid[deal_idx, river_idx] = 0
                    continue

                self.deal_river_valid[deal_idx, river_idx] = 1

                # 7 cards = 2 hole + 4 board + 1 river
                h0 = np.array(list(oop_hand) + board + [river_card], dtype=np.int32)
                h1 = np.array(list(ip_hand) + board + [river_card], dtype=np.int32)

                self.deal_river_p0_value[deal_idx, river_idx] = evaluate_7cards(h0)
                self.deal_river_p1_value[deal_idx, river_idx] = evaluate_7cards(h1)

        valid_total = self.deal_river_valid.sum()
        print(f"Precomputed {valid_total} valid (deal, river) hand values")

    def _setup_gpu_arrays(self):
        """Setup GPU arrays."""
        # Tree arrays on GPU
        self.node_type_gpu = cp.asarray(self.node_type)
        self.node_player_gpu = cp.asarray(self.node_player)
        self.node_num_actions_gpu = cp.asarray(self.node_num_actions)
        self.node_fold_player_gpu = cp.asarray(self.node_fold_player)
        self.node_children_gpu = cp.asarray(self.node_children.flatten())
        self.node_invested_oop_gpu = cp.asarray(self.node_invested_oop)
        self.node_invested_ip_gpu = cp.asarray(self.node_invested_ip)
        self.node_is_river_gpu = cp.asarray(self.node_is_river)

        # River hand values on GPU
        self.deal_river_valid_gpu = cp.asarray(self.deal_river_valid.flatten())
        self.deal_river_p0_value_gpu = cp.asarray(self.deal_river_p0_value.flatten())
        self.deal_river_p1_value_gpu = cp.asarray(self.deal_river_p1_value.flatten())

        # Infoset mapping
        node_h_p0 = np.full((self.num_nodes, self.num_deals), -1, dtype=np.int32)
        node_h_p1 = np.full((self.num_nodes, self.num_deals), -1, dtype=np.int32)

        for deal_idx in range(self.num_deals):
            oop_hand, ip_hand = self.game.get_deal(deal_idx)
            oop_hand = tuple(sorted(oop_hand, reverse=True))
            ip_hand = tuple(sorted(ip_hand, reverse=True))

            oop_hand_idx = self.oop_hand_to_idx.get(oop_hand, -1)
            ip_hand_idx = self.ip_hand_to_idx.get(ip_hand, -1)

            for node_idx, node in enumerate(self.nodes):
                if node['type'] == NODE_DECISION:
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

        # Regret and strategy arrays
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

    def _cfr_iteration_gpu(self):
        """Run one CFR iteration on GPU."""
        t = self.iterations + 1
        strat = self._compute_strategy_gpu()

        block_size = 128  # Smaller block size due to higher register usage
        grid_size = (self.num_deals + block_size - 1) // block_size
        iteration_weight = np.float32(t)

        num_possible_rivers = len(self.game.possible_rivers)

        TURN_CFR_KERNEL(
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
             self.node_is_river_gpu,
             np.int32(self.num_deals),
             np.int32(num_possible_rivers),
             self.deal_river_valid_gpu,
             self.deal_river_p0_value_gpu,
             self.deal_river_p1_value_gpu,
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

    def iterate(self, n: int = 1):
        """Run n CFR iterations."""
        for _ in range(n):
            self._cfr_iteration_gpu()
            self.iterations += 1

    def solve(self, iterations: int = 1000):
        """Solve for given number of iterations."""
        self.iterate(iterations)

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
        if node['type'] != NODE_DECISION:
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
        """Print strategies at key decision nodes."""
        from gpu_poker_cfr.games.cards import card_name

        for node_idx, node in enumerate(self.nodes):
            if node['type'] != NODE_DECISION:
                continue

            player = 'OOP' if node['player'] == 0 else 'IP'
            street = node.get('street', 'unknown')
            actions = node['actions']

            print(f"\n=== Node {node_idx}: {player} {street} (pot={node['pot']:.0f}) ===")
            print(f"Actions: {actions}")

            agg = self.get_aggregate_strategy(node_idx)
            print(f"Aggregate: {', '.join(f'{a}={v*100:.1f}%' for a, v in agg.items())}")


def test_turn_solver():
    """Quick test of the turn solver."""
    from gpu_poker_cfr.games.turn_toy_game import make_turn_toy_game
    import time

    print("Creating turn game...")
    game = make_turn_toy_game()

    print("\nCreating GPU turn solver...")
    solver = GPUTurnSolver(game)

    print("\nRunning CFR iterations...")
    start = time.time()
    solver.solve(iterations=1000)
    elapsed = time.time() - start

    print(f"\n1000 iterations in {elapsed:.2f}s ({1000/elapsed:.0f} iter/s)")

    print("\n=== Turn Root Strategy (OOP) ===")
    root_strat = solver.get_aggregate_strategy(0)
    for action, prob in root_strat.items():
        print(f"  {action}: {prob*100:.1f}%")


if __name__ == '__main__':
    test_turn_solver()
