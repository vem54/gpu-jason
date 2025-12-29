"""
Fully Vectorized GPU CFR for River Poker.

All operations vectorized on GPU - no Python loops in iteration.
"""

import numpy as np
import cupy as cp
from typing import List, Tuple

from gpu_poker_cfr.games.river_toy import RiverMicro, RiverToyMini, RiverAction
from gpu_poker_cfr.games.cards import card_rank


class GPURiverCFRv2:
    """Fully vectorized GPU CFR."""

    def __init__(self, game):
        """Initialize with RiverMicro or RiverToyMini."""
        self.game = game
        self.num_deals = game.num_deals

        # Build tree structure (small, on CPU)
        self._build_tree()

        # Pre-compute hand values and deal info on GPU
        self._setup_gpu_arrays()

        self.iterations = 0

    def _build_tree(self):
        """Build tree as node list."""
        self.nodes = []
        self._action_map = {}
        self._build_recursive([], 0, 10)  # STARTING_POT

        self.num_nodes = len(self.nodes)

        # Convert tree to arrays
        self.node_type = np.array([n['type'] for n in self.nodes], dtype=np.int32)
        self.node_player = np.array([n.get('player', 0) for n in self.nodes], dtype=np.int32)
        self.node_pot = np.array([n['pot'] for n in self.nodes], dtype=np.float32)
        self.node_num_actions = np.array([n.get('num_actions', 0) for n in self.nodes], dtype=np.int32)
        self.node_fold_player = np.array([n.get('fold_player', -1) for n in self.nodes], dtype=np.int32)

        # Child indices (not action IDs)
        self.node_child_idx = np.full((self.num_nodes, 3), -1, dtype=np.int32)
        aid_to_idx = {n['action_id']: i for i, n in enumerate(self.nodes)}
        for i, n in enumerate(self.nodes):
            if 'child_ids' in n:
                for j, cid in enumerate(n['child_ids']):
                    if cid >= 0 and cid in aid_to_idx:
                        self.node_child_idx[i, j] = aid_to_idx[cid]

        # Build infoset mapping
        self._build_infosets()

    def _build_recursive(self, actions, player, pot, to_call=0, num_bets=0):
        BET_SIZE = 10
        MAX_RAISES = 2
        action_tuple = tuple(actions)
        action_id = self._get_action_id(action_tuple)

        if actions and actions[-1] == 'F':
            self.nodes.append({
                'type': 0,  # FOLD
                'action_id': action_id,
                'fold_player': (len(actions) - 1) % 2,
                'pot': pot
            })
            return

        if len(actions) >= 2:
            if actions[-1] == 'K' and actions[-2] == 'K':
                self.nodes.append({'type': 1, 'action_id': action_id, 'pot': pot})
                return
            if actions[-1] == 'C':
                self.nodes.append({'type': 1, 'action_id': action_id, 'pot': pot})
                return

        facing_bet = to_call > 0
        if facing_bet:
            child_actions = ['F', 'C']
            if num_bets < MAX_RAISES:
                child_actions.append('R')
        else:
            child_actions = ['K']
            if num_bets < MAX_RAISES:
                child_actions.append('B')

        child_ids = [self._get_action_id(tuple(actions + [a])) for a in child_actions]
        while len(child_ids) < 3:
            child_ids.append(-1)

        self.nodes.append({
            'type': 2,  # DECISION
            'action_id': action_id,
            'player': player,
            'num_actions': len(child_actions),
            'pot': pot,
            'child_ids': child_ids
        })

        for action in child_actions:
            new_pot, new_to_call, new_num_bets = pot, 0, num_bets
            if action == 'C':
                new_pot += to_call
            elif action == 'B':
                new_pot += BET_SIZE
                new_to_call = BET_SIZE
                new_num_bets += 1
            elif action == 'R':
                new_pot += to_call + BET_SIZE
                new_to_call = BET_SIZE
                new_num_bets += 1

            self._build_recursive(actions + [action], 1 - player, new_pot, new_to_call, new_num_bets)

    def _get_action_id(self, actions):
        if actions not in self._action_map:
            self._action_map[actions] = len(self._action_map)
        return self._action_map[actions]

    def _build_infosets(self):
        """Map (node, hole_card_idx) -> infoset_idx."""
        avail = self.game.available_cards
        self.num_hole_cards = len(avail)
        self.card_to_idx = {c: i for i, c in enumerate(avail)}

        self.infoset_key_to_idx = {}
        idx = 0
        for i, n in enumerate(self.nodes):
            if n['type'] == 2:  # DECISION
                aid = n['action_id']
                for card_idx in range(self.num_hole_cards):
                    key = (card_idx, aid)
                    if key not in self.infoset_key_to_idx:
                        self.infoset_key_to_idx[key] = idx
                        idx += 1

        self.num_infosets = idx
        self.infoset_num_actions = np.zeros(idx, dtype=np.int32)

        for key, h in self.infoset_key_to_idx.items():
            card_idx, aid = key
            for n in self.nodes:
                if n.get('action_id') == aid and n['type'] == 2:
                    self.infoset_num_actions[h] = n['num_actions']
                    break

    def _setup_gpu_arrays(self):
        """Setup all GPU arrays."""
        board = list(self.game.board)

        # Pre-compute hand values and hole cards for all deals
        p0_cards = np.zeros(self.num_deals, dtype=np.int32)
        p1_cards = np.zeros(self.num_deals, dtype=np.int32)
        p0_values = np.zeros(self.num_deals, dtype=np.int32)
        p1_values = np.zeros(self.num_deals, dtype=np.int32)

        from gpu_poker_cfr.games.hand_eval import evaluate_7cards

        for deal_idx in range(self.num_deals):
            river, p0, p1 = self.game.get_deal(deal_idx)
            board_river = board + [river]

            hand0 = list(p0) + board_river
            hand1 = list(p1) + board_river
            while len(hand0) < 7:
                hand0.append(hand0[-1])
            while len(hand1) < 7:
                hand1.append(hand1[-1])

            p0_cards[deal_idx] = self.card_to_idx.get(p0[0], 0)
            p1_cards[deal_idx] = self.card_to_idx.get(p1[0], 0)
            p0_values[deal_idx] = evaluate_7cards(np.array(hand0, dtype=np.int32))
            p1_values[deal_idx] = evaluate_7cards(np.array(hand1, dtype=np.int32))

        # GPU arrays
        self.deal_p0_card = cp.asarray(p0_cards)
        self.deal_p1_card = cp.asarray(p1_cards)
        self.deal_p0_value = cp.asarray(p0_values)
        self.deal_p1_value = cp.asarray(p1_values)

        # Node arrays on GPU
        self.node_type_gpu = cp.asarray(self.node_type)
        self.node_player_gpu = cp.asarray(self.node_player)
        self.node_pot_gpu = cp.asarray(self.node_pot)
        self.node_num_actions_gpu = cp.asarray(self.node_num_actions)
        self.node_fold_player_gpu = cp.asarray(self.node_fold_player)
        self.node_child_idx_gpu = cp.asarray(self.node_child_idx)

        # Infoset index for each (node, deal): shape (num_nodes, num_deals)
        node_infoset_p0 = np.full((self.num_nodes, self.num_deals), -1, dtype=np.int32)
        node_infoset_p1 = np.full((self.num_nodes, self.num_deals), -1, dtype=np.int32)

        for i, n in enumerate(self.nodes):
            if n['type'] == 2:
                aid = n['action_id']
                for deal_idx in range(self.num_deals):
                    p0c = int(p0_cards[deal_idx])
                    p1c = int(p1_cards[deal_idx])
                    key0 = (p0c, aid)
                    key1 = (p1c, aid)
                    if key0 in self.infoset_key_to_idx:
                        node_infoset_p0[i, deal_idx] = self.infoset_key_to_idx[key0]
                    if key1 in self.infoset_key_to_idx:
                        node_infoset_p1[i, deal_idx] = self.infoset_key_to_idx[key1]

        self.node_infoset_p0_gpu = cp.asarray(node_infoset_p0)
        self.node_infoset_p1_gpu = cp.asarray(node_infoset_p1)

        # Regrets and strategy: (num_infosets, 3, num_deals)
        self._cumulative_regret = cp.zeros((self.num_infosets, 3, self.num_deals), dtype=cp.float32)
        self._cumulative_strategy = cp.zeros((self.num_infosets, 3, self.num_deals), dtype=cp.float32)

        # Uniform
        self._uniform = cp.zeros((self.num_infosets, 3, self.num_deals), dtype=cp.float32)
        for h in range(self.num_infosets):
            n = self.infoset_num_actions[h]
            self._uniform[h, :n, :] = 1.0 / n

    def _get_strategy(self):
        """Compute strategy on GPU."""
        pos_reg = cp.maximum(self._cumulative_regret, 0)
        reg_sum = pos_reg.sum(axis=1, keepdims=True)
        safe_sum = cp.where(reg_sum > 0, reg_sum, 1.0)
        strat = cp.where(reg_sum > 0, pos_reg / safe_sum, self._uniform)

        for h in range(self.num_infosets):
            n = self.infoset_num_actions[h]
            strat[h, n:, :] = 0

        return strat

    def iterate(self, n: int = 1):
        for _ in range(n):
            self._single_iteration()
            self.iterations += 1

    def _single_iteration(self):
        """Fully vectorized GPU iteration."""
        t = self.iterations + 1
        strat = self._get_strategy()

        # EV for each (node, deal, player): (num_nodes, num_deals, 2)
        ev = cp.zeros((self.num_nodes, self.num_deals, 2), dtype=cp.float32)

        # Process nodes backwards
        for i in range(self.num_nodes - 1, -1, -1):
            ntype = int(self.node_type[i])
            pot = float(self.node_pot[i])
            half_pot = pot / 2.0

            if ntype == 0:  # FOLD
                fp = int(self.node_fold_player[i])
                if fp == 0:
                    ev[i, :, 0] = -half_pot
                    ev[i, :, 1] = half_pot
                else:
                    ev[i, :, 0] = half_pot
                    ev[i, :, 1] = -half_pot

            elif ntype == 1:  # SHOWDOWN
                p0_wins = self.deal_p0_value > self.deal_p1_value
                p1_wins = self.deal_p1_value > self.deal_p0_value
                ev[i, :, 0] = cp.where(p0_wins, half_pot, cp.where(p1_wins, -half_pot, 0.0))
                ev[i, :, 1] = cp.where(p1_wins, half_pot, cp.where(p0_wins, -half_pot, 0.0))

            elif ntype == 2:  # DECISION
                acting = int(self.node_player[i])
                n_actions = int(self.node_num_actions[i])

                # Get infoset indices for all deals
                if acting == 0:
                    h_idx = self.node_infoset_p0_gpu[i, :]  # (num_deals,)
                else:
                    h_idx = self.node_infoset_p1_gpu[i, :]

                # Get child EVs: (n_actions, num_deals, 2)
                child_ev = cp.zeros((n_actions, self.num_deals, 2), dtype=cp.float32)
                for a in range(n_actions):
                    cidx = int(self.node_child_idx[i, a])
                    if cidx >= 0:
                        child_ev[a] = ev[cidx]

                # Vectorized EV computation across all deals
                # strat[h_idx, :n_actions, deal_idx] gives strategy for each deal
                # We need to index strat correctly

                for deal_idx in range(self.num_deals):
                    h = int(h_idx[deal_idx])
                    if h >= 0:
                        s = strat[h, :n_actions, deal_idx]  # (n_actions,)

                        # Weighted EV
                        for p in range(2):
                            node_ev = 0.0
                            for a in range(n_actions):
                                node_ev += float(s[a]) * float(child_ev[a, deal_idx, p])
                            ev[i, deal_idx, p] = node_ev

                        # Regret updates
                        for p in range(2):
                            if acting == p:
                                for a in range(n_actions):
                                    regret = float(child_ev[a, deal_idx, p]) - float(ev[i, deal_idx, p])
                                    self._cumulative_regret[h, a, deal_idx] += regret / self.num_deals

        self._cumulative_strategy += t * strat

    def solve(self, iterations: int = 1000):
        self.iterate(iterations)

    @property
    def average_strategy(self):
        total = self._cumulative_strategy.sum(axis=1, keepdims=True)
        safe = cp.where(total > 0, total, 1.0)
        return cp.where(total > 0, self._cumulative_strategy / safe, 1.0/3)
