"""
Array-based Semi-Vector MCCFR for Leduc Poker.

Convert game tree to dense arrays for vectorized processing.
Process all nodes in array operations instead of Python loops.
"""

import numpy as np
from typing import List, Tuple, Dict

from gpu_poker_cfr.games.base import Game
from gpu_poker_cfr.games.leduc import (
    LeducPoker, ANTE, ROUND1_BET, ROUND2_BET, MAX_RAISES
)


LEDUC_NUM_CARDS = 6
LEDUC_NUM_DEALS = 30
LEDUC_NUM_RANKS = 3

# Node types
NODE_TERMINAL_FOLD = 0
NODE_TERMINAL_SHOWDOWN = 1
NODE_TO_R2 = 2
NODE_DECISION = 3


class SemiVectorLeducArray:
    """Array-based Semi-Vector MCCFR for Leduc."""

    def __init__(self, game: Game, sample_boards: bool = False, num_samples: int = 1):
        if not isinstance(game, LeducPoker):
            raise ValueError("Only LeducPoker supported")

        self.game = game
        self.sample_boards = sample_boards
        self.num_samples = num_samples

        # Build array-based tree
        self._build_array_tree()

        # Regrets: (num_infosets, max_actions, num_deals)
        self._cumulative_regret = np.zeros(
            (self.num_infosets, 3, LEDUC_NUM_DEALS), dtype=np.float32
        )
        self._cumulative_strategy = np.zeros(
            (self.num_infosets, 3, LEDUC_NUM_DEALS), dtype=np.float32
        )

        # Pre-compute uniform strategy
        self._uniform = np.zeros((self.num_infosets, 3, LEDUC_NUM_DEALS), dtype=np.float32)
        for h in range(self.num_infosets):
            n = self.infoset_num_actions[h]
            self._uniform[h, :n, :] = 1.0 / n

        # Pre-compute scenarios
        self._build_scenarios()

        self.iterations = 0

    def _build_scenarios(self):
        """Pre-compute all deal/board scenarios."""
        scenarios = []
        for deal_idx in range(30):
            hero = deal_idx // 5
            offset = deal_idx % 5
            villain = offset if offset < hero else offset + 1

            remaining = [c for c in range(6) if c != hero and c != villain]
            for comm in remaining:
                hero_rank = hero // 2
                villain_rank = villain // 2
                comm_rank = comm // 2

                hero_pair = 1 if hero_rank == comm_rank else 0
                villain_pair = 1 if villain_rank == comm_rank else 0
                hero_strength = hero_pair * 100 + hero_rank
                villain_strength = villain_pair * 100 + villain_rank

                scenarios.append((deal_idx, hero_rank, villain_rank, comm_rank,
                                 hero_strength, villain_strength))

        self.scenarios = np.array(scenarios, dtype=np.int32)
        self.num_scenarios = len(scenarios)

    def _build_array_tree(self):
        """Build tree as dense arrays."""
        # First build as dicts to get structure
        self.r1_nodes = []
        self.r2_nodes = []
        self._action_map = {}
        self._build_round_tree(self.r1_nodes, 1, [], 0, ANTE, ANTE, 0)
        self._build_round_tree(self.r2_nodes, 2, [], 0, 0, 0, 0)

        # Convert to arrays
        self._convert_to_arrays()

        # Build infoset mapping
        self._build_infoset_mapping()

    def _build_round_tree(self, nodes: list, round_num: int, actions: list,
                          player: int, p0: int, p1: int, num_bets: int):
        bet_size = ROUND1_BET if round_num == 1 else ROUND2_BET
        action_tuple = tuple(actions)
        action_id = self._get_action_id(action_tuple)

        if actions and actions[-1] == 'f':
            fold_player = (len(actions) - 1) % 2
            nodes.append({
                'type': 'terminal_fold',
                'action_id': action_id,
                'fold_player': fold_player,
                'p0': p0, 'p1': p1,
                'actions': action_tuple
            })
            return

        round_over = False
        if len(actions) >= 2:
            if actions[-2:] == ['c', 'c']:
                round_over = True
            elif actions[-1] == 'c' and actions[-2] == 'b':
                round_over = True

        if round_over:
            nodes.append({
                'type': 'to_r2' if round_num == 1 else 'terminal_showdown',
                'action_id': action_id,
                'p0': p0, 'p1': p1,
                'actions': action_tuple
            })
            return

        facing_bet = self._facing_bet(actions)
        if facing_bet:
            num_actions = 3 if num_bets < MAX_RAISES else 2
        else:
            num_actions = 2 if num_bets < MAX_RAISES else 1

        # Build children
        if facing_bet:
            children = [actions + ['f'], actions + ['c']]
            if num_actions == 3:
                children.append(actions + ['b'])
        else:
            children = [actions + ['c']]
            if num_actions == 2:
                children.append(actions + ['b'])

        child_ids = [self._get_action_id(tuple(c)) for c in children]

        nodes.append({
            'type': 'decision',
            'action_id': action_id,
            'player': player,
            'facing_bet': facing_bet,
            'num_actions': num_actions,
            'p0': p0, 'p1': p1,
            'actions': action_tuple,
            'child_ids': child_ids
        })

        next_player = 1 - player

        if facing_bet:
            self._build_round_tree(nodes, round_num, actions + ['f'],
                                  next_player, p0, p1, num_bets)
            new_p0 = max(p0, p1) if player == 0 else p0
            new_p1 = max(p0, p1) if player == 1 else p1
            self._build_round_tree(nodes, round_num, actions + ['c'],
                                  next_player, new_p0, new_p1, num_bets)
            if num_bets < MAX_RAISES:
                if player == 0:
                    new_p0 = max(p0, p1) + bet_size
                else:
                    new_p1 = max(p0, p1) + bet_size
                self._build_round_tree(nodes, round_num, actions + ['b'],
                                      next_player,
                                      new_p0 if player == 0 else p0,
                                      new_p1 if player == 1 else p1,
                                      num_bets + 1)
        else:
            self._build_round_tree(nodes, round_num, actions + ['c'],
                                  next_player, p0, p1, num_bets)
            if num_bets < MAX_RAISES:
                if player == 0:
                    new_p0 = p0 + bet_size
                else:
                    new_p1 = p1 + bet_size
                self._build_round_tree(nodes, round_num, actions + ['b'],
                                      next_player,
                                      new_p0 if player == 0 else p0,
                                      new_p1 if player == 1 else p1,
                                      num_bets + 1)

    def _get_action_id(self, actions: tuple) -> int:
        if actions not in self._action_map:
            self._action_map[actions] = len(self._action_map)
        return self._action_map[actions]

    def _facing_bet(self, actions):
        if not actions:
            return False
        for i in range(len(actions) - 1, -1, -1):
            if actions[i] == 'b':
                return True
            if actions[i] == 'c' and i > 0 and actions[i-1] == 'b':
                return False
        return False

    def _convert_to_arrays(self):
        """Convert node lists to arrays."""
        # Round 1 arrays
        n1 = len(self.r1_nodes)
        self.r1_type = np.zeros(n1, dtype=np.int32)
        self.r1_action_id = np.zeros(n1, dtype=np.int32)
        self.r1_player = np.zeros(n1, dtype=np.int32)
        self.r1_num_actions = np.zeros(n1, dtype=np.int32)
        self.r1_p0 = np.zeros(n1, dtype=np.float32)
        self.r1_p1 = np.zeros(n1, dtype=np.float32)
        self.r1_fold_player = np.zeros(n1, dtype=np.int32)
        self.r1_child_ids = np.full((n1, 3), -1, dtype=np.int32)

        for i, node in enumerate(self.r1_nodes):
            self.r1_action_id[i] = node['action_id']
            self.r1_p0[i] = node['p0']
            self.r1_p1[i] = node['p1']

            if node['type'] == 'terminal_fold':
                self.r1_type[i] = NODE_TERMINAL_FOLD
                self.r1_fold_player[i] = node['fold_player']
            elif node['type'] == 'to_r2':
                self.r1_type[i] = NODE_TO_R2
            elif node['type'] == 'decision':
                self.r1_type[i] = NODE_DECISION
                self.r1_player[i] = node['player']
                self.r1_num_actions[i] = node['num_actions']
                for j, cid in enumerate(node['child_ids']):
                    self.r1_child_ids[i, j] = cid

        # Round 2 arrays
        n2 = len(self.r2_nodes)
        self.r2_type = np.zeros(n2, dtype=np.int32)
        self.r2_action_id = np.zeros(n2, dtype=np.int32)
        self.r2_player = np.zeros(n2, dtype=np.int32)
        self.r2_num_actions = np.zeros(n2, dtype=np.int32)
        self.r2_p0 = np.zeros(n2, dtype=np.float32)
        self.r2_p1 = np.zeros(n2, dtype=np.float32)
        self.r2_fold_player = np.zeros(n2, dtype=np.int32)
        self.r2_child_ids = np.full((n2, 3), -1, dtype=np.int32)

        for i, node in enumerate(self.r2_nodes):
            self.r2_action_id[i] = node['action_id']
            self.r2_p0[i] = node['p0']
            self.r2_p1[i] = node['p1']

            if node['type'] == 'terminal_fold':
                self.r2_type[i] = NODE_TERMINAL_FOLD
                self.r2_fold_player[i] = node['fold_player']
            elif node['type'] == 'terminal_showdown':
                self.r2_type[i] = NODE_TERMINAL_SHOWDOWN
            elif node['type'] == 'decision':
                self.r2_type[i] = NODE_DECISION
                self.r2_player[i] = node['player']
                self.r2_num_actions[i] = node['num_actions']
                for j, cid in enumerate(node['child_ids']):
                    self.r2_child_ids[i, j] = cid

        # Build action_id to node index mapping
        self.r1_aid_to_idx = {self.r1_action_id[i]: i for i in range(n1)}
        self.r2_aid_to_idx = {self.r2_action_id[i]: i for i in range(n2)}

    def _build_infoset_mapping(self):
        """Build infoset index mapping."""
        self.infoset_key_to_idx = {}
        self.infoset_info = []
        idx = 0

        # Round 1 infosets
        for i, node in enumerate(self.r1_nodes):
            if node['type'] == 'decision':
                aid = node['action_id']
                for rank in range(3):
                    key = (1, rank, -1, aid)
                    if key not in self.infoset_key_to_idx:
                        self.infoset_key_to_idx[key] = idx
                        self.infoset_info.append({
                            'player': node['player'],
                            'rank': rank,
                            'num_actions': node['num_actions']
                        })
                        idx += 1

        # Round 2 infosets
        for i, node in enumerate(self.r2_nodes):
            if node['type'] == 'decision':
                aid = node['action_id']
                for rank in range(3):
                    for comm in range(3):
                        key = (2, rank, comm, aid)
                        if key not in self.infoset_key_to_idx:
                            self.infoset_key_to_idx[key] = idx
                            self.infoset_info.append({
                                'player': node['player'],
                                'rank': rank,
                                'num_actions': node['num_actions']
                            })
                            idx += 1

        self.num_infosets = idx
        self.infoset_num_actions = np.array(
            [i['num_actions'] for i in self.infoset_info], dtype=np.int32
        )

    def _get_strategy_all(self) -> np.ndarray:
        """Get strategy for all infosets."""
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

        weight = 1.0 / 30 / 4

        # Process all scenarios
        for scenario in self.scenarios:
            deal_idx, hero_rank, villain_rank, comm_rank, hero_str, villain_str = scenario

            for player in range(2):
                player_rank = hero_rank if player == 0 else villain_rank
                self._update_scenario_array(
                    deal_idx, hero_rank, villain_rank, comm_rank,
                    hero_str, villain_str, player, player_rank,
                    strat, weight
                )

        self._cumulative_strategy += t * strat

    def _update_scenario_array(self, deal_idx, hero_rank, villain_rank, comm_rank,
                                hero_str, villain_str, player, player_rank,
                                strat, weight):
        """Update regrets for one scenario using array-based traversal."""
        # Round 1
        n1 = len(self.r1_nodes)
        r1_ev0 = np.zeros(n1, dtype=np.float32)
        r1_ev1 = np.zeros(n1, dtype=np.float32)

        # Process backwards (children before parents)
        for i in range(n1 - 1, -1, -1):
            node_type = self.r1_type[i]
            aid = self.r1_action_id[i]

            if node_type == NODE_TERMINAL_FOLD:
                fp = self.r1_fold_player[i]
                if fp == 0:
                    r1_ev0[i] = -self.r1_p0[i]
                    r1_ev1[i] = self.r1_p0[i]
                else:
                    r1_ev0[i] = self.r1_p1[i]
                    r1_ev1[i] = -self.r1_p1[i]

            elif node_type == NODE_TO_R2:
                ev = self._compute_r2_array(
                    hero_rank, villain_rank, comm_rank,
                    hero_str, villain_str,
                    self.r1_p0[i], self.r1_p1[i],
                    strat, deal_idx, player, player_rank, weight
                )
                r1_ev0[i] = ev[0]
                r1_ev1[i] = ev[1]

            elif node_type == NODE_DECISION:
                acting_player = self.r1_player[i]
                acting_rank = hero_rank if acting_player == 0 else villain_rank
                n_actions = self.r1_num_actions[i]

                # Get child EVs
                child_ev0 = []
                child_ev1 = []
                for j in range(n_actions):
                    cid = self.r1_child_ids[i, j]
                    if cid in self.r1_aid_to_idx:
                        cidx = self.r1_aid_to_idx[cid]
                        child_ev0.append(r1_ev0[cidx])
                        child_ev1.append(r1_ev1[cidx])
                    else:
                        child_ev0.append(0.0)
                        child_ev1.append(0.0)

                key = (1, acting_rank, -1, aid)
                if key in self.infoset_key_to_idx:
                    h_idx = self.infoset_key_to_idx[key]
                    s = strat[h_idx, :n_actions, deal_idx]

                    ev0 = sum(s[a] * child_ev0[a] for a in range(n_actions))
                    ev1 = sum(s[a] * child_ev1[a] for a in range(n_actions))
                    r1_ev0[i] = ev0
                    r1_ev1[i] = ev1

                    if acting_player == player and acting_rank == player_rank:
                        for a_idx in range(n_actions):
                            child_ev = child_ev0[a_idx] if player == 0 else child_ev1[a_idx]
                            node_ev = ev0 if player == 0 else ev1
                            self._cumulative_regret[h_idx, a_idx, deal_idx] += weight * (child_ev - node_ev)

    def _compute_r2_array(self, hero_rank, villain_rank, comm_rank,
                          hero_str, villain_str, p0_base, p1_base,
                          strat, deal_idx, player, player_rank, weight):
        """Compute round 2 EV."""
        player_ranks = [hero_rank, villain_rank]

        n2 = len(self.r2_nodes)
        r2_ev0 = np.zeros(n2, dtype=np.float32)
        r2_ev1 = np.zeros(n2, dtype=np.float32)

        for i in range(n2 - 1, -1, -1):
            node_type = self.r2_type[i]
            aid = self.r2_action_id[i]
            p0 = p0_base + self.r2_p0[i]
            p1 = p1_base + self.r2_p1[i]

            if node_type == NODE_TERMINAL_FOLD:
                fp = self.r2_fold_player[i]
                if fp == 0:
                    r2_ev0[i] = -p0
                    r2_ev1[i] = p0
                else:
                    r2_ev0[i] = p1
                    r2_ev1[i] = -p1

            elif node_type == NODE_TERMINAL_SHOWDOWN:
                if hero_str > villain_str:
                    r2_ev0[i] = p1
                    r2_ev1[i] = -p1
                elif villain_str > hero_str:
                    r2_ev0[i] = -p0
                    r2_ev1[i] = p0
                else:
                    r2_ev0[i] = 0.0
                    r2_ev1[i] = 0.0

            elif node_type == NODE_DECISION:
                acting_player = self.r2_player[i]
                acting_rank = player_ranks[acting_player]
                n_actions = self.r2_num_actions[i]

                child_ev0 = []
                child_ev1 = []
                for j in range(n_actions):
                    cid = self.r2_child_ids[i, j]
                    if cid in self.r2_aid_to_idx:
                        cidx = self.r2_aid_to_idx[cid]
                        child_ev0.append(r2_ev0[cidx])
                        child_ev1.append(r2_ev1[cidx])
                    else:
                        child_ev0.append(0.0)
                        child_ev1.append(0.0)

                key = (2, acting_rank, comm_rank, aid)
                if key in self.infoset_key_to_idx:
                    h_idx = self.infoset_key_to_idx[key]
                    s = strat[h_idx, :n_actions, deal_idx]

                    ev0 = sum(s[a] * child_ev0[a] for a in range(n_actions))
                    ev1 = sum(s[a] * child_ev1[a] for a in range(n_actions))
                    r2_ev0[i] = ev0
                    r2_ev1[i] = ev1

                    if acting_player == player and acting_rank == player_rank:
                        for a_idx in range(n_actions):
                            child_ev = child_ev0[a_idx] if player == 0 else child_ev1[a_idx]
                            node_ev = ev0 if player == 0 else ev1
                            self._cumulative_regret[h_idx, a_idx, deal_idx] += weight * (child_ev - node_ev)

        # Return root EV
        root_id = self._get_action_id(())
        if root_id in self.r2_aid_to_idx:
            root_idx = self.r2_aid_to_idx[root_id]
            return (r2_ev0[root_idx], r2_ev1[root_idx])
        return (0.0, 0.0)

    def solve(self, iterations: int = 1000):
        self.iterate(iterations)

    @property
    def average_strategy(self):
        total = self._cumulative_strategy.sum(axis=1, keepdims=True)
        safe = np.where(total > 0, total, 1.0)
        return np.where(total > 0, self._cumulative_strategy / safe, 1.0/3)
