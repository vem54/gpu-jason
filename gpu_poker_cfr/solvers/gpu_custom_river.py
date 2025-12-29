"""
GPU CFR Solver for Custom River Toy Game.

Handles:
- Check / All-in / Call / Fold actions
- Two-card hands
- Custom ranges per player
"""

import numpy as np
import cupy as cp
from typing import List, Tuple, Dict

from gpu_poker_cfr.games.hand_eval import evaluate_7cards


# CUDA kernel for CFR iteration with proper reach probability handling
# Regrets and strategies are stored PER INFOSET (not per deal)
CFR_KERNEL = cp.RawKernel(r'''
extern "C" __global__ void cfr_iteration(
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
    float* __restrict__ cumulative_strategy,

    const int num_infosets,
    const float iteration_weight
) {
    int deal = blockIdx.x * blockDim.x + threadIdx.x;
    if (deal >= num_deals) return;

    float ev0[32];
    float ev1[32];
    float reach0[32];  // Player 0 reach probability
    float reach1[32];  // Player 1 reach probability

    int p0_val = deal_p0_value[deal];
    int p1_val = deal_p1_value[deal];

    // Initialize root reach = 1
    reach0[0] = 1.0f;
    reach1[0] = 1.0f;

    // Forward pass: compute reach probabilities and accumulate strategy
    for (int i = 0; i < num_nodes; i++) {
        int ntype = node_type[i];
        if (ntype != 2) continue;  // Only decision nodes

        int acting = node_player[i];
        int n_act = node_num_actions[i];

        int h_idx;
        if (acting == 0) {
            h_idx = node_h_p0[i * num_deals + deal];
        } else {
            h_idx = node_h_p1[i * num_deals + deal];
        }

        int c0 = node_child_idx[i * 3 + 0];
        int c1 = node_child_idx[i * 3 + 1];
        int c2 = node_child_idx[i * 3 + 2];

        if (h_idx < 0) {
            // Invalid deal - set child reaches to 0
            if (c0 >= 0) { reach0[c0] = 0.0f; reach1[c0] = 0.0f; }
            if (c1 >= 0) { reach0[c1] = 0.0f; reach1[c1] = 0.0f; }
            if (c2 >= 0) { reach0[c2] = 0.0f; reach1[c2] = 0.0f; }
            continue;
        }

        // Strategy is per-infoset: strat[h_idx * 3 + action]
        int strat_base = h_idx * 3;
        float s0 = (n_act >= 1 && c0 >= 0) ? strat[strat_base + 0] : 0.0f;
        float s1 = (n_act >= 2 && c1 >= 0) ? strat[strat_base + 1] : 0.0f;
        float s2 = (n_act >= 3 && c2 >= 0) ? strat[strat_base + 2] : 0.0f;

        // Accumulate strategy weighted by player's own reach (per infoset)
        float own_reach = (acting == 0) ? reach0[i] : reach1[i];
        if (n_act >= 1) atomicAdd(&cumulative_strategy[strat_base + 0], s0 * own_reach * iteration_weight);
        if (n_act >= 2) atomicAdd(&cumulative_strategy[strat_base + 1], s1 * own_reach * iteration_weight);
        if (n_act >= 3) atomicAdd(&cumulative_strategy[strat_base + 2], s2 * own_reach * iteration_weight);

        // Propagate reach to children
        if (acting == 0) {
            if (c0 >= 0) { reach0[c0] = reach0[i] * s0; reach1[c0] = reach1[i]; }
            if (c1 >= 0) { reach0[c1] = reach0[i] * s1; reach1[c1] = reach1[i]; }
            if (c2 >= 0) { reach0[c2] = reach0[i] * s2; reach1[c2] = reach1[i]; }
        } else {
            if (c0 >= 0) { reach0[c0] = reach0[i]; reach1[c0] = reach1[i] * s0; }
            if (c1 >= 0) { reach0[c1] = reach0[i]; reach1[c1] = reach1[i] * s1; }
            if (c2 >= 0) { reach0[c2] = reach0[i]; reach1[c2] = reach1[i] * s2; }
        }
    }

    // Backward pass: compute EVs and update regrets
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
        else {  // DECISION
            int acting = node_player[i];
            int n_act = node_num_actions[i];

            int h_idx;
            if (acting == 0) {
                h_idx = node_h_p0[i * num_deals + deal];
            } else {
                h_idx = node_h_p1[i * num_deals + deal];
            }

            if (h_idx < 0) continue;

            int strat_base = h_idx * 3;
            int c0 = node_child_idx[i * 3 + 0];
            int c1 = node_child_idx[i * 3 + 1];
            int c2 = node_child_idx[i * 3 + 2];

            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
            float cev0_0 = 0.0f, cev0_1 = 0.0f, cev0_2 = 0.0f;
            float cev1_0 = 0.0f, cev1_1 = 0.0f, cev1_2 = 0.0f;

            if (n_act >= 1 && c0 >= 0) { s0 = strat[strat_base + 0]; cev0_0 = ev0[c0]; cev1_0 = ev1[c0]; }
            if (n_act >= 2 && c1 >= 0) { s1 = strat[strat_base + 1]; cev0_1 = ev0[c1]; cev1_1 = ev1[c1]; }
            if (n_act >= 3 && c2 >= 0) { s2 = strat[strat_base + 2]; cev0_2 = ev0[c2]; cev1_2 = ev1[c2]; }

            float node_ev0 = s0 * cev0_0 + s1 * cev0_1 + s2 * cev0_2;
            float node_ev1 = s0 * cev1_0 + s1 * cev1_1 + s2 * cev1_2;

            ev0[i] = node_ev0;
            ev1[i] = node_ev1;

            // Update regrets weighted by opponent reach (per infoset)
            int reg_base = h_idx * 3;
            float opp_reach = (acting == 0) ? reach1[i] : reach0[i];

            if (acting == 0) {
                if (n_act >= 1) atomicAdd(&cumulative_regret[reg_base + 0], (cev0_0 - node_ev0) * opp_reach);
                if (n_act >= 2) atomicAdd(&cumulative_regret[reg_base + 1], (cev0_1 - node_ev0) * opp_reach);
                if (n_act >= 3) atomicAdd(&cumulative_regret[reg_base + 2], (cev0_2 - node_ev0) * opp_reach);
            } else {
                if (n_act >= 1) atomicAdd(&cumulative_regret[reg_base + 0], (cev1_0 - node_ev1) * opp_reach);
                if (n_act >= 2) atomicAdd(&cumulative_regret[reg_base + 1], (cev1_1 - node_ev1) * opp_reach);
                if (n_act >= 3) atomicAdd(&cumulative_regret[reg_base + 2], (cev1_2 - node_ev1) * opp_reach);
            }
        }
    }
}
''', 'cfr_iteration')


# Strategy kernel - per infoset (not per deal)
STRATEGY_KERNEL = cp.RawKernel(r'''
extern "C" __global__ void compute_strategy(
    const float* __restrict__ cumulative_regret,
    float* __restrict__ strategy,
    const int* __restrict__ infoset_num_actions,
    const int num_infosets
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_infosets) return;

    int n_act = infoset_num_actions[h];
    int base = h * 3;

    float r0 = (n_act >= 1) ? fmaxf(cumulative_regret[base + 0], 0.0f) : 0.0f;
    float r1 = (n_act >= 2) ? fmaxf(cumulative_regret[base + 1], 0.0f) : 0.0f;
    float r2 = (n_act >= 3) ? fmaxf(cumulative_regret[base + 2], 0.0f) : 0.0f;

    float total = r0 + r1 + r2;

    if (total > 0.0f) {
        strategy[base + 0] = r0 / total;
        strategy[base + 1] = r1 / total;
        strategy[base + 2] = r2 / total;
    } else {
        float uniform = 1.0f / n_act;
        strategy[base + 0] = (n_act >= 1) ? uniform : 0.0f;
        strategy[base + 1] = (n_act >= 2) ? uniform : 0.0f;
        strategy[base + 2] = (n_act >= 3) ? uniform : 0.0f;
    }
}
''', 'compute_strategy')


class GPUCustomRiverCFR:
    """GPU CFR solver for custom river games."""

    def __init__(self, game):
        self.game = game
        self.num_deals = game.num_deals
        self.pot_size = game.pot_size
        self.stack_size = game.stack_size

        self._build_tree()
        self._setup_gpu_arrays()
        self.iterations = 0

    def _build_tree(self):
        """
        Build game tree for Check/All-in poker.

        Tree structure:
        - Root: OOP to act (Check or All-in)
        - After OOP Check: IP to act (Check or All-in)
        - After OOP Check, IP All-in: OOP to act (Fold or Call)
        - After OOP All-in: IP to act (Fold or Call)

        EV calculation notes:
        - pot stores the total pot including all bets
        - For showdown: winner gets pot/2, loser gets -pot/2 (relative to starting)
        - For fold: folder loses their initial investment (pot_size/2), opponent gains it
        """
        self.nodes = []
        self._action_map = {}

        pot = self.pot_size
        bet = self.stack_size
        initial_contribution = pot // 2  # What each player starts with in the pot

        # Node 0: Root - OOP to act, can Check or All-in
        self.nodes.append({
            'type': 2, 'action_id': 0, 'player': 0,
            'num_actions': 2, 'pot': pot,
            'actions': ['Check', 'All-in']
        })

        # Node 1: After OOP Check - IP to act
        self.nodes.append({
            'type': 2, 'action_id': 1, 'player': 1,
            'num_actions': 2, 'pot': pot,
            'actions': ['Check', 'All-in']
        })

        # Node 2: After Check-Check - Showdown
        self.nodes.append({
            'type': 1, 'action_id': 2, 'pot': pot
        })

        # Node 3: After Check-Allin - OOP to act (Fold/Call)
        self.nodes.append({
            'type': 2, 'action_id': 3, 'player': 0,
            'num_actions': 2, 'pot': pot + bet,  # IP bet into pot
            'actions': ['Fold', 'Call']
        })

        # Node 4: After Check-Allin-Fold - IP wins
        # OOP folds, loses their initial 50, IP wins 50
        self.nodes.append({
            'type': 0, 'action_id': 4, 'fold_player': 0,
            'pot': pot  # Use original pot for fold EV calculation
        })

        # Node 5: After Check-Allin-Call - Showdown
        self.nodes.append({
            'type': 1, 'action_id': 5, 'pot': pot + 2 * bet  # 300 total
        })

        # Node 6: After OOP All-in - IP to act (Fold/Call)
        self.nodes.append({
            'type': 2, 'action_id': 6, 'player': 1,
            'num_actions': 2, 'pot': pot + bet,  # OOP bet into pot
            'actions': ['Fold', 'Call']
        })

        # Node 7: After Allin-Fold - OOP wins
        # IP folds, loses their initial 50, OOP wins 50
        self.nodes.append({
            'type': 0, 'action_id': 7, 'fold_player': 1,
            'pot': pot  # Use original pot for fold EV calculation
        })

        # Node 8: After Allin-Call - Showdown
        self.nodes.append({
            'type': 1, 'action_id': 8, 'pot': pot + 2 * bet  # 300 total
        })

        self.num_nodes = len(self.nodes)

        # Build child indices
        # Node 0 (OOP: Check/Allin) -> Check:Node1, Allin:Node6
        # Node 1 (IP: Check/Allin) -> Check:Node2, Allin:Node3
        # Node 3 (OOP: Fold/Call) -> Fold:Node4, Call:Node5
        # Node 6 (IP: Fold/Call) -> Fold:Node7, Call:Node8
        self.node_child_idx = np.full((self.num_nodes, 3), -1, dtype=np.int32)
        self.node_child_idx[0] = [1, 6, -1]   # Check->1, Allin->6
        self.node_child_idx[1] = [2, 3, -1]   # Check->2, Allin->3
        self.node_child_idx[3] = [4, 5, -1]   # Fold->4, Call->5
        self.node_child_idx[6] = [7, 8, -1]   # Fold->7, Call->8

        # Convert to arrays
        self.node_type = np.array([n['type'] for n in self.nodes], dtype=np.int32)
        self.node_player = np.array([n.get('player', 0) for n in self.nodes], dtype=np.int32)
        self.node_pot = np.array([n['pot'] for n in self.nodes], dtype=np.float32)
        self.node_num_actions = np.array([n.get('num_actions', 0) for n in self.nodes], dtype=np.int32)
        self.node_fold_player = np.array([n.get('fold_player', -1) for n in self.nodes], dtype=np.int32)

        # Build infosets
        self._build_infosets()

    def _build_infosets(self):
        """Build infoset mapping based on player's hand and action history."""
        # OOP infosets: based on OOP hand
        # IP infosets: based on IP hand

        # Map (player, hand_idx, action_id) -> infoset_idx
        # Normalize hands by sorting (higher card first)
        self.oop_hands = list(set(tuple(sorted(h, reverse=True)) for h in self.game.oop_range))
        self.ip_hands = list(set(tuple(sorted(h, reverse=True)) for h in self.game.ip_range))

        self.oop_hand_to_idx = {h: i for i, h in enumerate(self.oop_hands)}
        self.ip_hand_to_idx = {h: i for i, h in enumerate(self.ip_hands)}

        # For each decision node, create infosets for the acting player's hands
        self.infoset_key_to_idx = {}
        idx = 0

        for node_idx, node in enumerate(self.nodes):
            if node['type'] == 2:  # Decision node
                player = node['player']
                action_id = node['action_id']

                if player == 0:  # OOP
                    for hand_idx in range(len(self.oop_hands)):
                        key = (player, hand_idx, action_id)
                        if key not in self.infoset_key_to_idx:
                            self.infoset_key_to_idx[key] = idx
                            idx += 1
                else:  # IP
                    for hand_idx in range(len(self.ip_hands)):
                        key = (player, hand_idx, action_id)
                        if key not in self.infoset_key_to_idx:
                            self.infoset_key_to_idx[key] = idx
                            idx += 1

        self.num_infosets = idx

        # Store number of actions per infoset
        self.infoset_num_actions = np.zeros(idx, dtype=np.int32)
        for key, h_idx in self.infoset_key_to_idx.items():
            player, hand_idx, action_id = key
            for node in self.nodes:
                if node.get('action_id') == action_id and node['type'] == 2:
                    self.infoset_num_actions[h_idx] = node['num_actions']
                    break

    def _setup_gpu_arrays(self):
        """Setup GPU arrays for CFR."""
        board = list(self.game.board)

        # Compute hand values for each deal
        p0_values = np.zeros(self.num_deals, dtype=np.int32)
        p1_values = np.zeros(self.num_deals, dtype=np.int32)

        for deal_idx in range(self.num_deals):
            oop_hand, ip_hand = self.game.get_deal(deal_idx)

            # Build 7-card hands
            h0 = list(oop_hand) + board
            h1 = list(ip_hand) + board

            p0_values[deal_idx] = evaluate_7cards(np.array(h0, dtype=np.int32))
            p1_values[deal_idx] = evaluate_7cards(np.array(h1, dtype=np.int32))

        self.deal_p0_value = cp.asarray(p0_values)
        self.deal_p1_value = cp.asarray(p1_values)

        # Map (node, deal) -> infoset index for each player
        node_h_p0 = np.full((self.num_nodes, self.num_deals), -1, dtype=np.int32)
        node_h_p1 = np.full((self.num_nodes, self.num_deals), -1, dtype=np.int32)

        for deal_idx in range(self.num_deals):
            oop_hand, ip_hand = self.game.get_deal(deal_idx)

            # Normalize hand (sort cards)
            oop_hand = tuple(sorted(oop_hand, reverse=True))
            ip_hand = tuple(sorted(ip_hand, reverse=True))

            oop_hand_idx = self.oop_hand_to_idx.get(oop_hand, -1)
            ip_hand_idx = self.ip_hand_to_idx.get(ip_hand, -1)

            for node_idx, node in enumerate(self.nodes):
                if node['type'] == 2:
                    player = node['player']
                    action_id = node['action_id']

                    if player == 0 and oop_hand_idx >= 0:
                        key = (0, oop_hand_idx, action_id)
                        if key in self.infoset_key_to_idx:
                            node_h_p0[node_idx, deal_idx] = self.infoset_key_to_idx[key]
                    elif player == 1 and ip_hand_idx >= 0:
                        key = (1, ip_hand_idx, action_id)
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
        self.node_child_idx_gpu = cp.asarray(self.node_child_idx.flatten())

        # Regret and strategy arrays - PER INFOSET (not per deal!)
        # Shape: (num_infosets, 3) for 3 actions max
        self._cumulative_regret = cp.zeros((self.num_infosets, 3), dtype=cp.float32)
        self._cumulative_strategy = cp.zeros((self.num_infosets, 3), dtype=cp.float32)
        self._strategy = cp.zeros((self.num_infosets, 3), dtype=cp.float32)

        self.infoset_num_actions_gpu = cp.asarray(self.infoset_num_actions)

    def _compute_strategy(self):
        block_size = 256
        grid_size = (self.num_infosets + block_size - 1) // block_size

        STRATEGY_KERNEL(
            (grid_size,), (block_size,),
            (self._cumulative_regret, self._strategy, self.infoset_num_actions_gpu,
             np.int32(self.num_infosets))
        )
        return self._strategy

    def iterate(self, n: int = 1):
        for _ in range(n):
            self._single_iteration()
            self.iterations += 1

    def _single_iteration(self):
        t = self.iterations + 1
        strat = self._compute_strategy()

        block_size = 256
        grid_size = (self.num_deals + block_size - 1) // block_size
        iteration_weight = np.float32(t)

        CFR_KERNEL(
            (grid_size,), (block_size,),
            (np.int32(self.num_nodes),
             self.node_type_gpu, self.node_player_gpu, self.node_pot_gpu,
             self.node_num_actions_gpu, self.node_fold_player_gpu, self.node_child_idx_gpu,
             np.int32(self.num_deals), self.deal_p0_value, self.deal_p1_value,
             self.node_h_p0, self.node_h_p1,
             strat, self._cumulative_regret, self._cumulative_strategy,
             np.int32(self.num_infosets), iteration_weight)
        )

        # CFR+ with regret floor of 0
        self._cumulative_regret = cp.maximum(self._cumulative_regret, 0)
        cp.cuda.Stream.null.synchronize()

    def solve(self, iterations: int = 1000):
        self.iterate(iterations)

    def get_strategy_for_hand(self, player: int, hand: Tuple[int, int], action_id: int) -> Dict[str, float]:
        """Get the average strategy for a specific hand at a decision point."""
        hand = tuple(sorted(hand, reverse=True))

        if player == 0:
            hand_idx = self.oop_hand_to_idx.get(hand, -1)
        else:
            hand_idx = self.ip_hand_to_idx.get(hand, -1)

        if hand_idx < 0:
            return {}

        key = (player, hand_idx, action_id)
        if key not in self.infoset_key_to_idx:
            return {}

        h = self.infoset_key_to_idx[key]

        # Get cumulative strategy for this infoset (per-infoset arrays)
        strat_sum = cp.asnumpy(self._cumulative_strategy[h, :])

        total = strat_sum.sum()
        if total > 0:
            probs = strat_sum / total
        else:
            n_act = self.infoset_num_actions[h]
            probs = np.array([1.0/n_act if i < n_act else 0 for i in range(3)])

        # Find action names
        for node in self.nodes:
            if node.get('action_id') == action_id and node['type'] == 2:
                actions = node.get('actions', ['Action0', 'Action1', 'Action2'])
                break
        else:
            actions = ['Action0', 'Action1', 'Action2']

        n_act = self.infoset_num_actions[h]
        result = {}
        for i in range(n_act):
            result[actions[i]] = float(probs[i])

        return result

    def print_oop_strategy(self):
        """Print OOP strategy at all decision points."""
        from gpu_poker_cfr.games.cards import card_name

        print("=" * 60)
        print("OOP Strategy (Out of Position Player)")
        print("=" * 60)

        # Node 0: Root - OOP first action
        print("\n--- At Root (OOP to act first) ---")
        print("Hand         | Check    | All-in")
        print("-" * 40)

        for hand in self.oop_hands:
            hand_str = f"{card_name(hand[0])}{card_name(hand[1])}"
            strat = self.get_strategy_for_hand(0, hand, action_id=0)
            check = strat.get('Check', 0) * 100
            allin = strat.get('All-in', 0) * 100
            print(f"{hand_str:12s} | {check:6.1f}%  | {allin:6.1f}%")

        # Node 3: After Check-Allin - OOP faces all-in
        print("\n--- After Check -> IP All-in (OOP faces all-in) ---")
        print("Hand         | Fold     | Call")
        print("-" * 40)

        for hand in self.oop_hands:
            hand_str = f"{card_name(hand[0])}{card_name(hand[1])}"
            strat = self.get_strategy_for_hand(0, hand, action_id=3)
            fold = strat.get('Fold', 0) * 100
            call = strat.get('Call', 0) * 100
            print(f"{hand_str:12s} | {fold:6.1f}%  | {call:6.1f}%")
