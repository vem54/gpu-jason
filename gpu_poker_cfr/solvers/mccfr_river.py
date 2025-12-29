"""
Monte Carlo CFR for River toy games.

Samples deals instead of full enumeration.
Uses Numba for fast tree traversal.
"""

import numpy as np
from numba import njit
from typing import Dict, List, Tuple, Optional

from gpu_poker_cfr.games.river_toy import RiverToyMini, RiverMicro, RiverAction, RiverState
from gpu_poker_cfr.games.cards import card_rank, card_name
from gpu_poker_cfr.games.hand_eval import evaluate_7cards


# Node types
NODE_TERMINAL_FOLD = 0
NODE_TERMINAL_SHOWDOWN = 1
NODE_DECISION = 2


@njit(cache=True)
def traverse_tree(
    num_nodes: int,
    node_type: np.ndarray,
    node_action_id: np.ndarray,
    node_player: np.ndarray,
    node_num_actions: np.ndarray,
    node_pot: np.ndarray,
    node_fold_player: np.ndarray,
    node_child_ids: np.ndarray,
    aid_to_idx: np.ndarray,
    infoset_idx_p0: np.ndarray,  # (num_nodes,) -> infoset idx for p0
    infoset_idx_p1: np.ndarray,  # (num_nodes,) -> infoset idx for p1
    p0_value: int,
    p1_value: int,
    strat: np.ndarray,  # (num_infosets, max_actions)
    player: int,  # Updating player
    regret_updates: np.ndarray,  # (num_infosets, max_actions)
    pi_0: float,  # P0's contribution to reach
    pi_1: float,  # P1's contribution to reach
):
    """Traverse tree and compute regret updates for one sampled deal."""
    ev0 = np.zeros(num_nodes, dtype=np.float32)
    ev1 = np.zeros(num_nodes, dtype=np.float32)

    # Process backwards
    for i in range(num_nodes - 1, -1, -1):
        ntype = node_type[i]
        pot = node_pot[i]
        half_pot = pot / 2.0

        if ntype == NODE_TERMINAL_FOLD:
            fp = node_fold_player[i]
            if fp == 0:
                ev0[i] = -half_pot
                ev1[i] = half_pot
            else:
                ev0[i] = half_pot
                ev1[i] = -half_pot

        elif ntype == NODE_TERMINAL_SHOWDOWN:
            if p0_value > p1_value:
                ev0[i] = half_pot
                ev1[i] = -half_pot
            elif p1_value > p0_value:
                ev0[i] = -half_pot
                ev1[i] = half_pot
            else:
                ev0[i] = 0.0
                ev1[i] = 0.0

        elif ntype == NODE_DECISION:
            acting = node_player[i]
            n_actions = node_num_actions[i]

            # Get infoset for acting player
            if acting == 0:
                h_idx = infoset_idx_p0[i]
            else:
                h_idx = infoset_idx_p1[i]

            if h_idx >= 0:
                s = strat[h_idx, :n_actions]

                # Child EVs
                child_ev0 = np.zeros(n_actions, dtype=np.float32)
                child_ev1 = np.zeros(n_actions, dtype=np.float32)

                for j in range(n_actions):
                    cid = node_child_ids[i, j]
                    if cid >= 0 and cid < len(aid_to_idx):
                        cidx = aid_to_idx[cid]
                        if cidx >= 0:
                            child_ev0[j] = ev0[cidx]
                            child_ev1[j] = ev1[cidx]

                # Compute weighted EV
                node_ev0 = 0.0
                node_ev1 = 0.0
                for a in range(n_actions):
                    node_ev0 += s[a] * child_ev0[a]
                    node_ev1 += s[a] * child_ev1[a]

                ev0[i] = node_ev0
                ev1[i] = node_ev1

                # Update regrets if this is updating player's node
                if acting == player:
                    # Counterfactual reach: opponent's contribution
                    cf_reach = pi_1 if acting == 0 else pi_0

                    for a_idx in range(n_actions):
                        if acting == 0:
                            regret = cf_reach * (child_ev0[a_idx] - node_ev0)
                        else:
                            regret = cf_reach * (child_ev1[a_idx] - node_ev1)
                        regret_updates[h_idx, a_idx] += regret


class MCCFRRiver:
    """Monte Carlo CFR for River games with sampling."""

    def __init__(self, game, num_buckets: int = 10):
        """Initialize with hand strength buckets.

        Args:
            game: RiverToyMini or RiverMicro
            num_buckets: Number of hand strength buckets
        """
        self.game = game
        self.num_buckets = num_buckets
        self.num_deals = game.num_deals

        # Build tree
        self._build_tree()

        # Compute hand strength buckets
        self._compute_buckets()

        # Regrets per infoset (bucket x action_seq)
        self._cumulative_regret = np.zeros(
            (self.num_infosets, 3), dtype=np.float32
        )
        self._cumulative_strategy = np.zeros(
            (self.num_infosets, 3), dtype=np.float32
        )

        # Uniform strategy
        self._uniform = np.zeros((self.num_infosets, 3), dtype=np.float32)
        for h in range(self.num_infosets):
            n = self.infoset_num_actions[h]
            self._uniform[h, :n] = 1.0 / n

        self.iterations = 0

    def _compute_buckets(self):
        """Compute hand strength buckets for each hole card combo."""
        # Get all unique hole card combinations
        hole_combos = set()
        for deal_idx in range(min(self.num_deals, 10000)):
            river, p0, p1 = self.game.get_deal(deal_idx)
            hole_combos.add(tuple(sorted(p0, reverse=True)))
            hole_combos.add(tuple(sorted(p1, reverse=True)))

        self.hole_combos = list(hole_combos)
        board = list(self.game.board)

        # Compute average hand strength for each combo
        strengths = []
        for combo in self.hole_combos:
            # Sample a few rivers for strength estimation
            total = 0.0
            count = 0
            for deal_idx in range(0, min(self.num_deals, 1000), 100):
                river, _, _ = self.game.get_deal(deal_idx)
                hand = list(combo) + board + [river]
                while len(hand) < 7:
                    hand.append(hand[-1])
                total += evaluate_7cards(np.array(hand, dtype=np.int32))
                count += 1
            strengths.append(total / max(count, 1))

        # Assign to buckets
        sorted_indices = np.argsort(strengths)
        bucket_size = len(sorted_indices) / self.num_buckets

        self.combo_to_bucket = {}
        for rank, idx in enumerate(sorted_indices):
            bucket = min(int(rank / bucket_size), self.num_buckets - 1)
            self.combo_to_bucket[self.hole_combos[idx]] = bucket

    def _build_tree(self):
        """Build tree as arrays for Numba."""
        nodes = []
        self._action_map = {}

        self._build_tree_recursive(nodes, [], 0, 10)

        # Convert to arrays
        n = len(nodes)
        self.num_nodes = n
        self.node_type = np.array([nd['type'] for nd in nodes], dtype=np.int32)
        self.node_action_id = np.array([nd['action_id'] for nd in nodes], dtype=np.int32)
        self.node_player = np.array([nd.get('player', 0) for nd in nodes], dtype=np.int32)
        self.node_num_actions = np.array([nd.get('num_actions', 0) for nd in nodes], dtype=np.int32)
        self.node_pot = np.array([nd['pot'] for nd in nodes], dtype=np.float32)
        self.node_fold_player = np.array([nd.get('fold_player', 0) for nd in nodes], dtype=np.int32)
        self.node_child_ids = np.array([nd.get('child_ids', [-1, -1, -1]) for nd in nodes], dtype=np.int32)

        # Action ID to node idx
        max_aid = max(self.node_action_id) + 1
        self.aid_to_idx = np.full(max_aid, -1, dtype=np.int32)
        for i, aid in enumerate(self.node_action_id):
            self.aid_to_idx[aid] = i

        # Build infoset mapping (bucket x action_seq)
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
                'pot': pot
            })
            return

        if len(actions) >= 2:
            if actions[-1] == RiverAction.CHECK and actions[-2] == RiverAction.CHECK:
                nodes.append({
                    'type': NODE_TERMINAL_SHOWDOWN,
                    'action_id': action_id,
                    'pot': pot
                })
                return
            if actions[-1] == RiverAction.CALL:
                nodes.append({
                    'type': NODE_TERMINAL_SHOWDOWN,
                    'action_id': action_id,
                    'pot': pot
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

        child_ids = []
        for action in child_actions:
            child_action_tuple = tuple(list(actions) + [action])
            child_id = self._get_action_id(child_action_tuple)
            child_ids.append(child_id)
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
        """Build infoset mapping: (bucket, action_id) -> infoset_idx."""
        self.infoset_key_to_idx = {}
        idx = 0

        # (node_idx, bucket) -> infoset for each player
        self.infoset_keys_p0 = np.full((len(nodes), self.num_buckets), -1, dtype=np.int32)
        self.infoset_keys_p1 = np.full((len(nodes), self.num_buckets), -1, dtype=np.int32)

        for i, nd in enumerate(nodes):
            if nd['type'] == NODE_DECISION:
                aid = nd['action_id']
                for bucket in range(self.num_buckets):
                    key = (bucket, aid)
                    if key not in self.infoset_key_to_idx:
                        self.infoset_key_to_idx[key] = idx
                        idx += 1
                    h_idx = self.infoset_key_to_idx[key]
                    self.infoset_keys_p0[i, bucket] = h_idx
                    self.infoset_keys_p1[i, bucket] = h_idx

        self.num_infosets = idx
        self.infoset_num_actions = np.zeros(idx, dtype=np.int32)

        for key, h_idx in self.infoset_key_to_idx.items():
            bucket, aid = key
            node_idx = self.aid_to_idx[aid]
            self.infoset_num_actions[h_idx] = self.node_num_actions[node_idx]

    def _get_bucket(self, hole_cards: tuple) -> int:
        """Get bucket for hole cards."""
        sorted_cards = tuple(sorted(hole_cards, reverse=True))
        return self.combo_to_bucket.get(sorted_cards, 0)

    def _get_strategy(self):
        """Compute current strategy from regrets."""
        pos_reg = np.maximum(self._cumulative_regret, 0)
        reg_sum = pos_reg.sum(axis=1, keepdims=True)

        safe_sum = np.where(reg_sum > 0, reg_sum, 1.0)
        strat = np.where(reg_sum > 0, pos_reg / safe_sum, self._uniform)

        for h in range(self.num_infosets):
            n = self.infoset_num_actions[h]
            strat[h, n:] = 0

        return strat

    def iterate(self, n: int = 1, samples_per_iter: int = 100):
        """Run n iterations with sampling."""
        for _ in range(n):
            self._single_iteration(samples_per_iter)
            self.iterations += 1

    def _single_iteration(self, num_samples: int):
        """Single MCCFR iteration with sampling."""
        t = self.iterations + 1
        strat = self._get_strategy()

        regret_updates = np.zeros_like(self._cumulative_regret)
        board = list(self.game.board)

        # Sample deals
        sample_indices = np.random.randint(0, self.num_deals, num_samples)

        for deal_idx in sample_indices:
            river, p0, p1 = self.game.get_deal(deal_idx)

            # Compute hand values
            hand0 = list(p0) + board + [river]
            hand1 = list(p1) + board + [river]
            while len(hand0) < 7:
                hand0.append(hand0[-1])
            while len(hand1) < 7:
                hand1.append(hand1[-1])

            p0_value = evaluate_7cards(np.array(hand0, dtype=np.int32))
            p1_value = evaluate_7cards(np.array(hand1, dtype=np.int32))

            # Get buckets
            p0_bucket = self._get_bucket(p0)
            p1_bucket = self._get_bucket(p1)

            # Get infoset indices for this deal
            infoset_idx_p0 = self.infoset_keys_p0[:, p0_bucket]
            infoset_idx_p1 = self.infoset_keys_p1[:, p1_bucket]

            for player in range(2):
                traverse_tree(
                    self.num_nodes,
                    self.node_type, self.node_action_id, self.node_player,
                    self.node_num_actions, self.node_pot, self.node_fold_player,
                    self.node_child_ids, self.aid_to_idx,
                    infoset_idx_p0, infoset_idx_p1,
                    p0_value, p1_value,
                    strat, player, regret_updates,
                    1.0, 1.0
                )

        self._cumulative_regret += regret_updates / num_samples
        self._cumulative_strategy += t * strat

    def solve(self, iterations: int = 1000, samples_per_iter: int = 100):
        self.iterate(iterations, samples_per_iter)

    @property
    def average_strategy(self):
        total = self._cumulative_strategy.sum(axis=1, keepdims=True)
        safe = np.where(total > 0, total, 1.0)
        return np.where(total > 0, self._cumulative_strategy / safe, 1.0/3)
