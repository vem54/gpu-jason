"""
GPU Semi-Vector CFR for River Poker.

All computation on GPU using CuPy:
- Regrets and strategies stored on GPU
- Hand evaluation batched on GPU
- EV computation vectorized across all deals
- Regret updates batched on GPU
"""

import numpy as np
import cupy as cp
from typing import List, Tuple, Optional

from gpu_poker_cfr.games.river_toy import RiverMicro, RiverToyMini, RiverAction
from gpu_poker_cfr.games.cards import card_rank


# Node types
NODE_TERMINAL_FOLD = 0
NODE_TERMINAL_SHOWDOWN = 1
NODE_DECISION = 2


# GPU kernel for 7-card hand evaluation
HAND_EVAL_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void evaluate_hands(
    const int* hands,      // (num_hands, 7) - card indices
    int* values,           // (num_hands,) - output hand values
    int num_hands
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_hands) return;

    const int* hand = hands + idx * 7;

    // Count ranks and suits
    int rank_counts[13] = {0};
    int suit_counts[4] = {0};
    int rank_mask = 0;

    for (int i = 0; i < 7; i++) {
        int card = hand[i];
        int rank = card / 4;
        int suit = card % 4;
        rank_counts[rank]++;
        suit_counts[suit]++;
        rank_mask |= (1 << rank);
    }

    // Check flush
    int flush_suit = -1;
    for (int s = 0; s < 4; s++) {
        if (suit_counts[s] >= 5) {
            flush_suit = s;
            break;
        }
    }

    // Check straight (including wheel A-2-3-4-5)
    int straight_high = -1;
    int wheel = 0x100F;  // A,2,3,4,5 = bits 12,3,2,1,0
    if ((rank_mask & wheel) == wheel) {
        straight_high = 3;  // 5-high
    } else {
        for (int high = 12; high >= 4; high--) {
            int mask = 0x1F << (high - 4);  // 5 consecutive bits
            if ((rank_mask & mask) == mask) {
                straight_high = high;
                break;
            }
        }
    }

    // Find quads, trips, pairs
    int quads_rank = -1, trips_rank = -1;
    int pair1 = -1, pair2 = -1;
    int kickers[5];
    int num_kickers = 0;

    for (int r = 12; r >= 0; r--) {
        int c = rank_counts[r];
        if (c == 4 && quads_rank < 0) quads_rank = r;
        else if (c == 3 && trips_rank < 0) trips_rank = r;
        else if (c == 3 && pair1 < 0) pair1 = r;
        else if (c == 2 && pair1 < 0) pair1 = r;
        else if (c == 2 && pair2 < 0) pair2 = r;
        else if (c == 1 && num_kickers < 5) kickers[num_kickers++] = r;
    }

    int value = 0;

    // Straight flush
    if (flush_suit >= 0 && straight_high >= 0) {
        // Check if straight uses flush cards
        int flush_rank_mask = 0;
        for (int i = 0; i < 7; i++) {
            if (hand[i] % 4 == flush_suit) {
                flush_rank_mask |= (1 << (hand[i] / 4));
            }
        }
        int sf_high = -1;
        if ((flush_rank_mask & wheel) == wheel) sf_high = 3;
        for (int high = 12; high >= 4 && sf_high < 0; high--) {
            int mask = 0x1F << (high - 4);
            if ((flush_rank_mask & mask) == mask) sf_high = high;
        }
        if (sf_high >= 0) {
            value = (8 << 20) | sf_high;
            values[idx] = value;
            return;
        }
    }

    // Four of a kind
    if (quads_rank >= 0) {
        int kicker = (trips_rank >= 0) ? trips_rank :
                     (pair1 >= 0) ? pair1 :
                     (num_kickers > 0) ? kickers[0] : 0;
        value = (7 << 20) | (quads_rank << 4) | kicker;
        values[idx] = value;
        return;
    }

    // Full house
    if (trips_rank >= 0 && pair1 >= 0) {
        value = (6 << 20) | (trips_rank << 4) | pair1;
        values[idx] = value;
        return;
    }

    // Flush
    if (flush_suit >= 0) {
        int flush_ranks[7];
        int fc = 0;
        for (int i = 0; i < 7 && fc < 5; i++) {
            if (hand[i] % 4 == flush_suit) {
                flush_ranks[fc++] = hand[i] / 4;
            }
        }
        // Sort descending (simple bubble)
        for (int i = 0; i < fc; i++) {
            for (int j = i+1; j < fc; j++) {
                if (flush_ranks[j] > flush_ranks[i]) {
                    int t = flush_ranks[i];
                    flush_ranks[i] = flush_ranks[j];
                    flush_ranks[j] = t;
                }
            }
        }
        value = (5 << 20);
        for (int i = 0; i < 5 && i < fc; i++) {
            value |= flush_ranks[i] << (16 - i*4);
        }
        values[idx] = value;
        return;
    }

    // Straight
    if (straight_high >= 0) {
        value = (4 << 20) | straight_high;
        values[idx] = value;
        return;
    }

    // Three of a kind
    if (trips_rank >= 0) {
        value = (3 << 20) | (trips_rank << 8);
        for (int i = 0; i < 2 && i < num_kickers; i++) {
            value |= kickers[i] << (4 - i*4);
        }
        values[idx] = value;
        return;
    }

    // Two pair
    if (pair1 >= 0 && pair2 >= 0) {
        value = (2 << 20) | (pair1 << 8) | (pair2 << 4);
        if (num_kickers > 0) value |= kickers[0];
        values[idx] = value;
        return;
    }

    // One pair
    if (pair1 >= 0) {
        value = (1 << 20) | (pair1 << 12);
        for (int i = 0; i < 3 && i < num_kickers; i++) {
            value |= kickers[i] << (8 - i*4);
        }
        values[idx] = value;
        return;
    }

    // High card
    value = 0;
    for (int i = 0; i < 5 && i < num_kickers; i++) {
        value |= kickers[i] << (16 - i*4);
    }
    values[idx] = value;
}
''', 'evaluate_hands')


class GPURiverCFR:
    """GPU-accelerated CFR for River poker."""

    def __init__(self, game: RiverMicro):
        """Initialize with game.

        Args:
            game: RiverMicro or similar game with get_deal() method
        """
        self.game = game
        self.num_deals = game.num_deals

        # Build tree structure (on CPU, it's small)
        self._build_tree()

        # Pre-compute all hand values on GPU
        self._compute_hand_values_gpu()

        # Regrets on GPU: (num_infosets, max_actions, num_deals)
        self._cumulative_regret = cp.zeros(
            (self.num_infosets, 3, self.num_deals), dtype=cp.float32
        )
        self._cumulative_strategy = cp.zeros(
            (self.num_infosets, 3, self.num_deals), dtype=cp.float32
        )

        # Pre-compute uniform strategy on GPU
        self._uniform = cp.zeros((self.num_infosets, 3, self.num_deals), dtype=cp.float32)
        for h in range(self.num_infosets):
            n = self.infoset_num_actions[h]
            self._uniform[h, :n, :] = 1.0 / n

        # Pre-compute infoset indices for all deals on GPU
        self._precompute_infoset_indices()

        self.iterations = 0

    def _compute_hand_values_gpu(self):
        """Compute hand values for all deals on GPU."""
        board = list(self.game.board)

        # Build hands array: (num_deals * 2, 7)
        hands = []
        for deal_idx in range(self.num_deals):
            river, p0, p1 = self.game.get_deal(deal_idx)
            board_river = board + [river]

            hand0 = list(p0) + board_river
            hand1 = list(p1) + board_river

            # Pad to 7 cards
            while len(hand0) < 7:
                hand0.append(hand0[-1])
            while len(hand1) < 7:
                hand1.append(hand1[-1])

            hands.append(hand0)
            hands.append(hand1)

        hands_np = np.array(hands, dtype=np.int32)
        hands_gpu = cp.asarray(hands_np)

        # Evaluate on GPU
        num_hands = len(hands)
        values_gpu = cp.zeros(num_hands, dtype=cp.int32)

        block_size = 256
        grid_size = (num_hands + block_size - 1) // block_size

        HAND_EVAL_KERNEL((grid_size,), (block_size,),
                         (hands_gpu, values_gpu, num_hands))

        # Split into p0 and p1 values
        values = values_gpu.reshape(self.num_deals, 2)
        self.deal_p0_value = values[:, 0]  # GPU array
        self.deal_p1_value = values[:, 1]  # GPU array

        # Also store hole card indices
        self.deal_p0_card = cp.zeros(self.num_deals, dtype=cp.int32)
        self.deal_p1_card = cp.zeros(self.num_deals, dtype=cp.int32)

        for deal_idx in range(self.num_deals):
            river, p0, p1 = self.game.get_deal(deal_idx)
            self.deal_p0_card[deal_idx] = p0[0]
            self.deal_p1_card[deal_idx] = p1[0]

    def _build_tree(self):
        """Build tree structure."""
        nodes = []
        self._action_map = {}
        self._build_tree_recursive(nodes, [], 0, 10)  # STARTING_POT

        self.num_nodes = len(nodes)
        self.nodes = nodes

        # Build infoset mapping
        self._build_infosets(nodes)

    def _build_tree_recursive(self, nodes, actions, player, pot, to_call=0, num_bets=0):
        action_tuple = tuple(actions)
        action_id = self._get_action_id(action_tuple)
        BET_SIZE = 10
        MAX_RAISES = 2

        if actions and actions[-1] == RiverAction.FOLD:
            fold_player = (len(actions) - 1) % 2
            nodes.append({
                'type': NODE_TERMINAL_FOLD,
                'action_id': action_id,
                'fold_player': fold_player,
                'pot': pot,
                'player': player
            })
            return

        if len(actions) >= 2:
            if actions[-1] == RiverAction.CHECK and actions[-2] == RiverAction.CHECK:
                nodes.append({
                    'type': NODE_TERMINAL_SHOWDOWN,
                    'action_id': action_id,
                    'pot': pot,
                    'player': player
                })
                return
            if actions[-1] == RiverAction.CALL:
                nodes.append({
                    'type': NODE_TERMINAL_SHOWDOWN,
                    'action_id': action_id,
                    'pot': pot,
                    'player': player
                })
                return

        facing_bet = to_call > 0
        if facing_bet:
            num_actions = 3 if num_bets < MAX_RAISES else 2
            child_actions = [RiverAction.FOLD, RiverAction.CALL]
            if num_bets < MAX_RAISES:
                child_actions.append(RiverAction.RAISE)
        else:
            num_actions = 2 if num_bets < MAX_RAISES else 1
            child_actions = [RiverAction.CHECK]
            if num_bets < MAX_RAISES:
                child_actions.append(RiverAction.BET)

        child_ids = [self._get_action_id(tuple(list(actions) + [a])) for a in child_actions]
        while len(child_ids) < 3:
            child_ids.append(-1)

        nodes.append({
            'type': NODE_DECISION,
            'action_id': action_id,
            'player': player,
            'num_actions': num_actions,
            'pot': pot,
            'child_ids': child_ids
        })

        next_player = 1 - player
        for action in child_actions:
            new_pot = pot
            new_to_call = 0
            new_num_bets = num_bets

            if action == RiverAction.CALL:
                new_pot += to_call
            elif action == RiverAction.BET:
                new_pot += BET_SIZE
                new_to_call = BET_SIZE
                new_num_bets += 1
            elif action == RiverAction.RAISE:
                new_pot += to_call + BET_SIZE
                new_to_call = BET_SIZE
                new_num_bets += 1

            self._build_tree_recursive(
                nodes, list(actions) + [action],
                next_player, new_pot, new_to_call, new_num_bets
            )

    def _get_action_id(self, actions):
        if actions not in self._action_map:
            self._action_map[actions] = len(self._action_map)
        return self._action_map[actions]

    def _build_infosets(self, nodes):
        """Build infoset mapping: (card, action_id) -> infoset_idx."""
        self.infoset_key_to_idx = {}
        idx = 0

        avail = self.game.available_cards
        self.card_to_idx = {c: i for i, c in enumerate(avail)}
        num_cards = len(avail)

        for node in nodes:
            if node['type'] == NODE_DECISION:
                aid = node['action_id']
                for card in avail:
                    card_idx = self.card_to_idx[card]
                    key = (card_idx, aid)
                    if key not in self.infoset_key_to_idx:
                        self.infoset_key_to_idx[key] = idx
                        idx += 1

        self.num_infosets = idx
        self.infoset_num_actions = np.zeros(idx, dtype=np.int32)

        for key, h_idx in self.infoset_key_to_idx.items():
            card_idx, aid = key
            for node in nodes:
                if node.get('action_id') == aid and node['type'] == NODE_DECISION:
                    self.infoset_num_actions[h_idx] = node['num_actions']
                    break

    def _precompute_infoset_indices(self):
        """Pre-compute infoset indices for all deals."""
        # For each (node, deal), store the infoset index
        # Shape: (num_nodes, num_deals)
        self.node_infoset_p0 = cp.full((self.num_nodes, self.num_deals), -1, dtype=cp.int32)
        self.node_infoset_p1 = cp.full((self.num_nodes, self.num_deals), -1, dtype=cp.int32)

        for i, node in enumerate(self.nodes):
            if node['type'] == NODE_DECISION:
                aid = node['action_id']

                # For each deal
                for deal_idx in range(self.num_deals):
                    p0_card = int(self.deal_p0_card[deal_idx])
                    p1_card = int(self.deal_p1_card[deal_idx])

                    p0_card_idx = self.card_to_idx.get(p0_card, 0)
                    p1_card_idx = self.card_to_idx.get(p1_card, 0)

                    key0 = (p0_card_idx, aid)
                    key1 = (p1_card_idx, aid)

                    if key0 in self.infoset_key_to_idx:
                        self.node_infoset_p0[i, deal_idx] = self.infoset_key_to_idx[key0]
                    if key1 in self.infoset_key_to_idx:
                        self.node_infoset_p1[i, deal_idx] = self.infoset_key_to_idx[key1]

    def _get_strategy_gpu(self) -> cp.ndarray:
        """Compute strategy from regrets on GPU."""
        pos_reg = cp.maximum(self._cumulative_regret, 0)
        reg_sum = pos_reg.sum(axis=1, keepdims=True)

        safe_sum = cp.where(reg_sum > 0, reg_sum, 1.0)
        strat = cp.where(reg_sum > 0, pos_reg / safe_sum, self._uniform)

        # Zero out invalid actions
        for h in range(self.num_infosets):
            n = self.infoset_num_actions[h]
            strat[h, n:, :] = 0

        return strat

    def iterate(self, n: int = 1):
        """Run n iterations."""
        for _ in range(n):
            self._single_iteration_gpu()
            self.iterations += 1

    def _single_iteration_gpu(self):
        """Single CFR iteration on GPU."""
        t = self.iterations + 1
        strat = self._get_strategy_gpu()

        # Process tree and compute regret updates
        # EVs: (num_nodes, num_deals, 2) for p0 and p1
        ev = cp.zeros((self.num_nodes, self.num_deals, 2), dtype=cp.float32)

        # Process nodes backwards (leaves to root)
        for i in range(self.num_nodes - 1, -1, -1):
            node = self.nodes[i]
            pot = node['pot']
            half_pot = pot / 2.0

            if node['type'] == NODE_TERMINAL_FOLD:
                fp = node['fold_player']
                if fp == 0:
                    ev[i, :, 0] = -half_pot
                    ev[i, :, 1] = half_pot
                else:
                    ev[i, :, 0] = half_pot
                    ev[i, :, 1] = -half_pot

            elif node['type'] == NODE_TERMINAL_SHOWDOWN:
                # Compare hand values (GPU arrays)
                p0_wins = self.deal_p0_value > self.deal_p1_value
                p1_wins = self.deal_p1_value > self.deal_p0_value

                ev[i, :, 0] = cp.where(p0_wins, half_pot,
                              cp.where(p1_wins, -half_pot, 0.0))
                ev[i, :, 1] = cp.where(p1_wins, half_pot,
                              cp.where(p0_wins, -half_pot, 0.0))

            elif node['type'] == NODE_DECISION:
                acting = node['player']
                n_actions = node['num_actions']
                child_ids = node['child_ids']

                # Get infoset indices for all deals
                if acting == 0:
                    h_indices = self.node_infoset_p0[i, :]  # (num_deals,)
                else:
                    h_indices = self.node_infoset_p1[i, :]

                # Get child EVs
                child_ev = cp.zeros((n_actions, self.num_deals, 2), dtype=cp.float32)
                for a in range(n_actions):
                    cid = child_ids[a]
                    if cid >= 0:
                        # Find node with this action_id
                        for j, n2 in enumerate(self.nodes):
                            if n2['action_id'] == cid:
                                child_ev[a] = ev[j]
                                break

                # Compute weighted EV for each deal
                for deal_idx in range(self.num_deals):
                    h_idx = int(h_indices[deal_idx])
                    if h_idx >= 0:
                        s = strat[h_idx, :n_actions, deal_idx]

                        for p in range(2):
                            node_ev = 0.0
                            for a in range(n_actions):
                                node_ev += float(s[a]) * float(child_ev[a, deal_idx, p])
                            ev[i, deal_idx, p] = node_ev

                        # Update regrets for updating player
                        for player in range(2):
                            if acting == player:
                                for a in range(n_actions):
                                    regret = float(child_ev[a, deal_idx, player]) - float(ev[i, deal_idx, player])
                                    self._cumulative_regret[h_idx, a, deal_idx] += regret / self.num_deals

        # Update cumulative strategy
        self._cumulative_strategy += t * strat

    def solve(self, iterations: int = 1000):
        self.iterate(iterations)

    @property
    def average_strategy(self):
        total = self._cumulative_strategy.sum(axis=1, keepdims=True)
        safe = cp.where(total > 0, total, 1.0)
        return cp.where(total > 0, self._cumulative_strategy / safe, 1.0/3)

    def get_average_strategy_cpu(self):
        return cp.asnumpy(self.average_strategy)
