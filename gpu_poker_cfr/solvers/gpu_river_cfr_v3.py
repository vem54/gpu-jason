"""
Truly Vectorized GPU CFR for River Poker.

NO Python loops over deals - everything vectorized with CuPy.
"""

import numpy as np
import cupy as cp
from typing import List, Tuple

from gpu_poker_cfr.games.river_toy import RiverMicro, RiverToyMini, RiverAction
from gpu_poker_cfr.games.hand_eval import evaluate_7cards


class GPURiverCFRv3:
    """Truly vectorized GPU CFR - no Python loops over deals."""

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
        self.deal_p0_card = cp.asarray(p0_cards)
        self.deal_p1_card = cp.asarray(p1_cards)
        self.deal_p0_value = cp.asarray(p0_values)
        self.deal_p1_value = cp.asarray(p1_values)

        # Node infoset indices per deal: (num_nodes, num_deals)
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

        # Regrets: (num_infosets, 3, num_deals)
        self._cumulative_regret = cp.zeros((self.num_infosets, 3, self.num_deals), dtype=cp.float32)
        self._cumulative_strategy = cp.zeros((self.num_infosets, 3, self.num_deals), dtype=cp.float32)

        # Uniform
        self._uniform = cp.zeros((self.num_infosets, 3, self.num_deals), dtype=cp.float32)
        for h in range(self.num_infosets):
            n = self.infoset_num_actions[h]
            self._uniform[h, :n, :] = 1.0 / n

        # Node arrays on GPU
        self.node_child_idx_gpu = cp.asarray(self.node_child_idx)

    def _get_strategy(self):
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
        """Vectorized iteration - minimal Python loops."""
        t = self.iterations + 1
        strat = self._get_strategy()  # (num_infosets, 3, num_deals)

        # EV: (num_nodes, num_deals, 2)
        ev = cp.zeros((self.num_nodes, self.num_deals, 2), dtype=cp.float32)

        # Process nodes backwards (this loop is O(num_nodes) ~ 15, not O(num_deals))
        for i in range(self.num_nodes - 1, -1, -1):
            ntype = int(self.node_type[i])
            pot = float(self.node_pot[i])
            half_pot = pot / 2.0

            if ntype == 0:  # FOLD - vectorized across deals
                fp = int(self.node_fold_player[i])
                ev[i, :, 0] = -half_pot if fp == 0 else half_pot
                ev[i, :, 1] = half_pot if fp == 0 else -half_pot

            elif ntype == 1:  # SHOWDOWN - vectorized across deals
                p0_wins = self.deal_p0_value > self.deal_p1_value
                p1_wins = self.deal_p1_value > self.deal_p0_value
                ev[i, :, 0] = cp.where(p0_wins, half_pot, cp.where(p1_wins, -half_pot, 0.0))
                ev[i, :, 1] = cp.where(p1_wins, half_pot, cp.where(p0_wins, -half_pot, 0.0))

            elif ntype == 2:  # DECISION
                acting = int(self.node_player[i])
                n_act = int(self.node_num_actions[i])

                # Infoset indices: (num_deals,)
                h_idx = self.node_h_p0[i] if acting == 0 else self.node_h_p1[i]

                # Child EVs: (n_act, num_deals, 2)
                child_ev = cp.zeros((n_act, self.num_deals, 2), dtype=cp.float32)
                for a in range(n_act):
                    cidx = int(self.node_child_idx[i, a])
                    if cidx >= 0:
                        child_ev[a] = ev[cidx]

                # Get strategy for each deal using advanced indexing
                # strat shape: (num_infosets, 3, num_deals)
                # We need strat[h_idx[d], :n_act, d] for each deal d

                # Build strategy tensor for this node: (n_act, num_deals)
                s = cp.zeros((n_act, self.num_deals), dtype=cp.float32)
                valid = h_idx >= 0
                for a in range(n_act):
                    # Gather strategy values
                    s[a, valid] = strat[h_idx[valid], a, cp.arange(self.num_deals)[valid]]

                # Compute weighted EV: sum over actions
                # child_ev: (n_act, num_deals, 2), s: (n_act, num_deals)
                for p in range(2):
                    ev[i, :, p] = cp.sum(s * child_ev[:, :, p], axis=0)

                # Update regrets - vectorized scatter add
                valid_mask = h_idx >= 0
                if cp.any(valid_mask):
                    deals = cp.arange(self.num_deals)
                    for p in range(2):
                        if acting == p:
                            for a in range(n_act):
                                regret = (child_ev[a, :, p] - ev[i, :, p]) / self.num_deals
                                # Use advanced indexing for scatter-add
                                h_valid = h_idx[valid_mask]
                                d_valid = deals[valid_mask]
                                r_valid = regret[valid_mask]
                                # Accumulate using cupyx.scatter_add
                                import cupyx
                                cupyx.scatter_add(
                                    self._cumulative_regret,
                                    (h_valid, a, d_valid),
                                    r_valid
                                )

        self._cumulative_strategy += t * strat

    def solve(self, iterations: int = 1000):
        self.iterate(iterations)

    @property
    def average_strategy(self):
        total = self._cumulative_strategy.sum(axis=1, keepdims=True)
        safe = cp.where(total > 0, total, 1.0)
        return cp.where(total > 0, self._cumulative_strategy / safe, 1.0/3)
