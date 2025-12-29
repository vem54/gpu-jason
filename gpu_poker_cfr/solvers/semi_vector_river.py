"""
Semi-Vector MCCFR for River toy games.

Vectorizes over all deals while processing tree nodes.
Uses Numba for fast tree traversal.
"""

import numpy as np
from numba import njit
from typing import Dict, List, Tuple

from gpu_poker_cfr.games.river_toy import RiverMicro, RiverToyMini, RiverAction, RiverState
from gpu_poker_cfr.games.cards import card_rank
from gpu_poker_cfr.games.hand_eval import evaluate_7cards


# Node types
NODE_TERMINAL_FOLD = 0
NODE_TERMINAL_SHOWDOWN = 1
NODE_DECISION = 2


@njit(cache=True)
def process_tree_nodes(
    num_nodes: int,
    node_type: np.ndarray,
    node_action_id: np.ndarray,
    node_player: np.ndarray,
    node_num_actions: np.ndarray,
    node_pot: np.ndarray,
    node_fold_player: np.ndarray,
    node_child_ids: np.ndarray,
    aid_to_idx: np.ndarray,
    infoset_keys: np.ndarray,  # (num_nodes, num_hole_cards) -> infoset idx or -1
    hero_card: int,
    villain_card: int,
    hero_value: int,
    villain_value: int,
    strat: np.ndarray,
    deal_idx: int,
    player: int,
    weight: float,
    regret_updates: np.ndarray
):
    """Process all tree nodes for one deal."""
    ev0 = np.zeros(num_nodes, dtype=np.float32)
    ev1 = np.zeros(num_nodes, dtype=np.float32)

    # Process backwards (leaves to root)
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
            if hero_value > villain_value:
                ev0[i] = half_pot
                ev1[i] = -half_pot
            elif villain_value > hero_value:
                ev0[i] = -half_pot
                ev1[i] = half_pot
            else:
                ev0[i] = 0.0
                ev1[i] = 0.0

        elif ntype == NODE_DECISION:
            acting_player = node_player[i]
            acting_card = hero_card if acting_player == 0 else villain_card
            n_actions = node_num_actions[i]

            # Get child EVs
            child_ev0 = np.zeros(3, dtype=np.float32)
            child_ev1 = np.zeros(3, dtype=np.float32)
            for j in range(n_actions):
                cid = node_child_ids[i, j]
                if cid >= 0 and cid < len(aid_to_idx):
                    cidx = aid_to_idx[cid]
                    if cidx >= 0:
                        child_ev0[j] = ev0[cidx]
                        child_ev1[j] = ev1[cidx]

            # Get infoset index
            h_idx = infoset_keys[i, acting_card]
            if h_idx >= 0:
                s = strat[h_idx, :n_actions, deal_idx]

                node_ev0 = 0.0
                node_ev1 = 0.0
                for a in range(n_actions):
                    node_ev0 += s[a] * child_ev0[a]
                    node_ev1 += s[a] * child_ev1[a]

                ev0[i] = node_ev0
                ev1[i] = node_ev1

                # Update regrets for current player
                player_card = hero_card if player == 0 else villain_card
                if acting_player == player and acting_card == player_card:
                    for a_idx in range(n_actions):
                        if player == 0:
                            child_ev = child_ev0[a_idx]
                            node_ev = node_ev0
                        else:
                            child_ev = child_ev1[a_idx]
                            node_ev = node_ev1
                        regret_updates[h_idx, a_idx, deal_idx] += weight * (child_ev - node_ev)


class SemiVectorRiver:
    """Semi-Vector MCCFR for River toy game."""

    def __init__(self, game: RiverMicro):
        self.game = game
        self.num_deals = game.num_deals

        # Build tree structure
        self._build_tree()

        # Pre-compute hand values for all deals
        self._compute_hand_values()

        # Regrets: (num_infosets, max_actions, num_deals)
        self._cumulative_regret = np.zeros(
            (self.num_infosets, 3, self.num_deals), dtype=np.float32
        )
        self._cumulative_strategy = np.zeros(
            (self.num_infosets, 3, self.num_deals), dtype=np.float32
        )

        # Uniform strategy
        self._uniform = np.zeros((self.num_infosets, 3, self.num_deals), dtype=np.float32)
        for h in range(self.num_infosets):
            n = self.infoset_num_actions[h]
            self._uniform[h, :n, :] = 1.0 / n

        self.iterations = 0

    def _compute_hand_values(self):
        """Pre-compute hand values for all deals."""
        self.deal_hero_value = np.zeros(self.num_deals, dtype=np.int32)
        self.deal_villain_value = np.zeros(self.num_deals, dtype=np.int32)
        self.deal_hero_card = np.zeros(self.num_deals, dtype=np.int32)
        self.deal_villain_card = np.zeros(self.num_deals, dtype=np.int32)

        board = list(self.game.board)

        for deal_idx in range(self.num_deals):
            river, p0, p1 = self.game.get_deal(deal_idx)
            board_river = board + [river]

            # Build 7-card hands (pad with duplicates if needed)
            hand0 = list(p0) + board_river
            hand1 = list(p1) + board_river
            while len(hand0) < 7:
                hand0.append(hand0[-1])
            while len(hand1) < 7:
                hand1.append(hand1[-1])

            self.deal_hero_value[deal_idx] = evaluate_7cards(np.array(hand0, dtype=np.int32))
            self.deal_villain_value[deal_idx] = evaluate_7cards(np.array(hand1, dtype=np.int32))
            self.deal_hero_card[deal_idx] = p0[0]
            self.deal_villain_card[deal_idx] = p1[0]

    def _build_tree(self):
        """Build tree as arrays for Numba."""
        nodes = []
        self._action_map = {}

        # Build tree recursively, then flatten
        self._build_tree_recursive(nodes, [], 0, self.game.board, 10)  # STARTING_POT

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

        # Build action_id -> node_idx mapping
        max_aid = max(self.node_action_id) + 1
        self.aid_to_idx = np.full(max_aid, -1, dtype=np.int32)
        for i, aid in enumerate(self.node_action_id):
            self.aid_to_idx[aid] = i

        # Build infoset mapping
        self._build_infosets(nodes)

    def _build_tree_recursive(self, nodes, actions, player, board, pot, to_call=0, num_bets=0):
        """Build tree recursively."""
        action_tuple = tuple(actions)
        action_id = self._get_action_id(action_tuple)
        BET_SIZE = 10
        MAX_RAISES = 2

        # Check terminal conditions
        if actions and actions[-1] == RiverAction.FOLD:
            fold_player = (len(actions) - 1) % 2
            nodes.append({
                'type': NODE_TERMINAL_FOLD,
                'action_id': action_id,
                'fold_player': fold_player,
                'pot': pot
            })
            return

        # Check-check or bet-call ends
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

        # Decision node
        facing_bet = to_call > 0
        if facing_bet:
            num_actions = 3 if num_bets < MAX_RAISES else 2  # FOLD, CALL, [RAISE]
            child_actions = [RiverAction.FOLD, RiverAction.CALL]
            if num_bets < MAX_RAISES:
                child_actions.append(RiverAction.RAISE)
        else:
            num_actions = 2 if num_bets < MAX_RAISES else 1  # CHECK, [BET]
            child_actions = [RiverAction.CHECK]
            if num_bets < MAX_RAISES:
                child_actions.append(RiverAction.BET)

        # Build children first to get their action IDs
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

        # Recurse for children
        next_player = 1 - player
        for action in child_actions:
            new_pot = pot
            new_to_call = 0
            new_num_bets = num_bets

            if action == RiverAction.FOLD:
                pass  # Handled above
            elif action == RiverAction.CALL:
                new_pot += to_call
            elif action == RiverAction.CHECK:
                pass
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
                next_player, board, new_pot, new_to_call, new_num_bets
            )

    def _get_action_id(self, actions):
        if actions not in self._action_map:
            self._action_map[actions] = len(self._action_map)
        return self._action_map[actions]

    def _build_infosets(self, nodes):
        """Build infoset mapping."""
        self.infoset_key_to_idx = {}
        idx = 0

        # Map hole card to index (0-3)
        avail = self.game.available_cards
        self.card_to_idx = {c: i for i, c in enumerate(avail)}
        num_cards = len(avail)

        # (node_idx, card_idx) -> infoset_idx
        self.infoset_keys = np.full((len(nodes), num_cards), -1, dtype=np.int32)

        for i, nd in enumerate(nodes):
            if nd['type'] == NODE_DECISION:
                aid = nd['action_id']
                for card in avail:
                    card_idx = self.card_to_idx[card]
                    key = (card_idx, aid)
                    if key not in self.infoset_key_to_idx:
                        self.infoset_key_to_idx[key] = idx
                        self.infoset_keys[i, card_idx] = idx
                        idx += 1
                    else:
                        self.infoset_keys[i, card_idx] = self.infoset_key_to_idx[key]

        self.num_infosets = idx

        # Store num_actions per infoset
        self.infoset_num_actions = np.zeros(idx, dtype=np.int32)
        for key, h_idx in self.infoset_key_to_idx.items():
            card_idx, aid = key
            node_idx = self.aid_to_idx[aid]
            self.infoset_num_actions[h_idx] = self.node_num_actions[node_idx]

    def _get_strategy_all(self):
        """Compute current strategy from regrets."""
        pos_reg = np.maximum(self._cumulative_regret, 0)
        reg_sum = pos_reg.sum(axis=1, keepdims=True)

        safe_sum = np.where(reg_sum > 0, reg_sum, 1.0)
        strat = np.where(reg_sum > 0, pos_reg / safe_sum, self._uniform)

        for h in range(self.num_infosets):
            n = self.infoset_num_actions[h]
            strat[h, n:, :] = 0

        return strat

    def iterate(self, n: int = 1):
        for _ in range(n):
            self._single_iteration()
            self.iterations += 1

    def _single_iteration(self):
        t = self.iterations + 1
        strat = self._get_strategy_all()

        weight = 1.0 / self.num_deals
        regret_updates = np.zeros_like(self._cumulative_regret)

        # Process all deals
        for deal_idx in range(self.num_deals):
            hero_card = self.deal_hero_card[deal_idx]
            villain_card = self.deal_villain_card[deal_idx]
            hero_value = self.deal_hero_value[deal_idx]
            villain_value = self.deal_villain_value[deal_idx]

            # Map cards to indices
            hero_idx = self.card_to_idx.get(hero_card, 0)
            villain_idx = self.card_to_idx.get(villain_card, 0)

            for player in range(2):
                process_tree_nodes(
                    self.num_nodes,
                    self.node_type, self.node_action_id, self.node_player,
                    self.node_num_actions, self.node_pot, self.node_fold_player,
                    self.node_child_ids, self.aid_to_idx, self.infoset_keys,
                    hero_idx, villain_idx, hero_value, villain_value,
                    strat, deal_idx, player, weight, regret_updates
                )

        self._cumulative_regret += regret_updates
        self._cumulative_strategy += t * strat

    def solve(self, iterations: int = 1000):
        self.iterate(iterations)

    @property
    def average_strategy(self):
        total = self._cumulative_strategy.sum(axis=1, keepdims=True)
        safe = np.where(total > 0, total, 1.0)
        return np.where(total > 0, self._cumulative_strategy / safe, 1.0/3)
