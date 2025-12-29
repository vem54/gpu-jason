"""
Numba-accelerated Semi-Vector MCCFR for Leduc Poker.

Uses Numba JIT compilation for the tree traversal loops.
"""

import numpy as np
from numba import njit, prange
from typing import Dict

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


@njit(cache=True)
def process_r1_node_terminal_fold(fold_player: int, p0: float, p1: float):
    """Process terminal fold node."""
    if fold_player == 0:
        return -p0, p0
    else:
        return p1, -p1


@njit(cache=True)
def process_r2_showdown(hero_str: int, villain_str: int, p0: float, p1: float):
    """Process showdown node."""
    if hero_str > villain_str:
        return p1, -p1
    elif villain_str > hero_str:
        return -p0, p0
    else:
        return 0.0, 0.0


@njit(cache=True)
def compute_ev_from_children(strategy: np.ndarray, child_ev0: np.ndarray,
                              child_ev1: np.ndarray, n_actions: int):
    """Compute weighted EV from child nodes."""
    ev0 = 0.0
    ev1 = 0.0
    for a in range(n_actions):
        ev0 += strategy[a] * child_ev0[a]
        ev1 += strategy[a] * child_ev1[a]
    return ev0, ev1


@njit(cache=True, parallel=False)
def process_round2_nodes(
    num_nodes: int,
    node_type: np.ndarray,
    node_action_id: np.ndarray,
    node_player: np.ndarray,
    node_num_actions: np.ndarray,
    node_p0: np.ndarray,
    node_p1: np.ndarray,
    node_fold_player: np.ndarray,
    node_child_ids: np.ndarray,
    aid_to_idx: np.ndarray,  # Dense mapping: aid -> node_idx
    infoset_keys: np.ndarray,  # (num_nodes, 3, 4) -> infoset idx or -1
    hero_rank: int,
    villain_rank: int,
    comm_rank: int,
    hero_str: int,
    villain_str: int,
    p0_base: float,
    p1_base: float,
    strat: np.ndarray,
    deal_idx: int,
    player: int,
    player_rank: int,
    weight: float,
    regret_updates: np.ndarray
):
    """Process all round 2 nodes."""
    player_ranks = np.array([hero_rank, villain_rank], dtype=np.int32)

    ev0 = np.zeros(num_nodes, dtype=np.float32)
    ev1 = np.zeros(num_nodes, dtype=np.float32)

    # Process backwards
    for i in range(num_nodes - 1, -1, -1):
        ntype = node_type[i]
        p0 = p0_base + node_p0[i]
        p1 = p1_base + node_p1[i]

        if ntype == NODE_TERMINAL_FOLD:
            fp = node_fold_player[i]
            if fp == 0:
                ev0[i] = -p0
                ev1[i] = p0
            else:
                ev0[i] = p1
                ev1[i] = -p1

        elif ntype == NODE_TERMINAL_SHOWDOWN:
            if hero_str > villain_str:
                ev0[i] = p1
                ev1[i] = -p1
            elif villain_str > hero_str:
                ev0[i] = -p0
                ev1[i] = p0
            else:
                ev0[i] = 0.0
                ev1[i] = 0.0

        elif ntype == NODE_DECISION:
            acting_player = node_player[i]
            acting_rank = player_ranks[acting_player]
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
            h_idx = infoset_keys[i, acting_rank, comm_rank]
            if h_idx >= 0:
                s = strat[h_idx, :n_actions, deal_idx]

                node_ev0 = 0.0
                node_ev1 = 0.0
                for a in range(n_actions):
                    node_ev0 += s[a] * child_ev0[a]
                    node_ev1 += s[a] * child_ev1[a]

                ev0[i] = node_ev0
                ev1[i] = node_ev1

                # Update regrets
                if acting_player == player and acting_rank == player_rank:
                    for a_idx in range(n_actions):
                        if player == 0:
                            child_ev = child_ev0[a_idx]
                            node_ev = node_ev0
                        else:
                            child_ev = child_ev1[a_idx]
                            node_ev = node_ev1
                        regret_updates[h_idx, a_idx, deal_idx] += weight * (child_ev - node_ev)

    return ev0[0], ev1[0]  # Return root EV


@njit(cache=True, parallel=False)
def process_round1_nodes(
    num_nodes: int,
    node_type: np.ndarray,
    node_action_id: np.ndarray,
    node_player: np.ndarray,
    node_num_actions: np.ndarray,
    node_p0: np.ndarray,
    node_p1: np.ndarray,
    node_fold_player: np.ndarray,
    node_child_ids: np.ndarray,
    aid_to_idx: np.ndarray,
    infoset_keys: np.ndarray,  # (num_nodes, 3) -> infoset idx or -1
    # R2 arrays
    r2_num_nodes: int,
    r2_node_type: np.ndarray,
    r2_node_action_id: np.ndarray,
    r2_node_player: np.ndarray,
    r2_node_num_actions: np.ndarray,
    r2_node_p0: np.ndarray,
    r2_node_p1: np.ndarray,
    r2_node_fold_player: np.ndarray,
    r2_node_child_ids: np.ndarray,
    r2_aid_to_idx: np.ndarray,
    r2_infoset_keys: np.ndarray,
    # Scenario info
    hero_rank: int,
    villain_rank: int,
    comm_rank: int,
    hero_str: int,
    villain_str: int,
    strat: np.ndarray,
    deal_idx: int,
    player: int,
    player_rank: int,
    weight: float,
    regret_updates: np.ndarray
):
    """Process all round 1 nodes."""
    ev0 = np.zeros(num_nodes, dtype=np.float32)
    ev1 = np.zeros(num_nodes, dtype=np.float32)

    for i in range(num_nodes - 1, -1, -1):
        ntype = node_type[i]

        if ntype == NODE_TERMINAL_FOLD:
            fp = node_fold_player[i]
            if fp == 0:
                ev0[i] = -node_p0[i]
                ev1[i] = node_p0[i]
            else:
                ev0[i] = node_p1[i]
                ev1[i] = -node_p1[i]

        elif ntype == NODE_TO_R2:
            # Process R2
            r2_ev = process_round2_nodes(
                r2_num_nodes,
                r2_node_type, r2_node_action_id, r2_node_player,
                r2_node_num_actions, r2_node_p0, r2_node_p1,
                r2_node_fold_player, r2_node_child_ids, r2_aid_to_idx,
                r2_infoset_keys,
                hero_rank, villain_rank, comm_rank, hero_str, villain_str,
                node_p0[i], node_p1[i],
                strat, deal_idx, player, player_rank, weight,
                regret_updates
            )
            ev0[i] = r2_ev[0]
            ev1[i] = r2_ev[1]

        elif ntype == NODE_DECISION:
            acting_player = node_player[i]
            acting_rank = hero_rank if acting_player == 0 else villain_rank
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

            h_idx = infoset_keys[i, acting_rank]
            if h_idx >= 0:
                s = strat[h_idx, :n_actions, deal_idx]

                node_ev0 = 0.0
                node_ev1 = 0.0
                for a in range(n_actions):
                    node_ev0 += s[a] * child_ev0[a]
                    node_ev1 += s[a] * child_ev1[a]

                ev0[i] = node_ev0
                ev1[i] = node_ev1

                if acting_player == player and acting_rank == player_rank:
                    for a_idx in range(n_actions):
                        if player == 0:
                            child_ev = child_ev0[a_idx]
                            node_ev = node_ev0
                        else:
                            child_ev = child_ev1[a_idx]
                            node_ev = node_ev1
                        regret_updates[h_idx, a_idx, deal_idx] += weight * (child_ev - node_ev)


class SemiVectorLeducNumba:
    """Numba-accelerated Semi-Vector MCCFR for Leduc."""

    def __init__(self, game: Game, sample_boards: bool = False, num_samples: int = 1):
        if not isinstance(game, LeducPoker):
            raise ValueError("Only LeducPoker supported")

        self.game = game
        self.sample_boards = sample_boards
        self.num_samples = num_samples

        # Build tree
        self._build_tree()

        # Regrets
        self._cumulative_regret = np.zeros(
            (self.num_infosets, 3, LEDUC_NUM_DEALS), dtype=np.float32
        )
        self._cumulative_strategy = np.zeros(
            (self.num_infosets, 3, LEDUC_NUM_DEALS), dtype=np.float32
        )

        # Uniform fallback
        self._uniform = np.zeros((self.num_infosets, 3, LEDUC_NUM_DEALS), dtype=np.float32)
        for h in range(self.num_infosets):
            n = self.infoset_num_actions[h]
            self._uniform[h, :n, :] = 1.0 / n

        # Pre-compute scenarios
        self._build_scenarios()

        self.iterations = 0

    def _build_scenarios(self):
        """Pre-compute scenarios."""
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

    def _build_tree(self):
        """Build tree as arrays for Numba."""
        self._action_map = {}
        r1_nodes = []
        r2_nodes = []
        self._build_round_tree(r1_nodes, 1, [], 0, ANTE, ANTE, 0)
        self._build_round_tree(r2_nodes, 2, [], 0, 0, 0, 0)

        # Convert to arrays
        self._convert_r1(r1_nodes)
        self._convert_r2(r2_nodes)
        self._build_infosets(r1_nodes, r2_nodes)

    def _build_round_tree(self, nodes, round_num, actions, player, p0, p1, num_bets):
        bet_size = ROUND1_BET if round_num == 1 else ROUND2_BET
        action_tuple = tuple(actions)
        action_id = self._get_action_id(action_tuple)

        if actions and actions[-1] == 'f':
            fold_player = (len(actions) - 1) % 2
            nodes.append({
                'type': NODE_TERMINAL_FOLD,
                'action_id': action_id,
                'fold_player': fold_player,
                'p0': p0, 'p1': p1,
                'player': 0, 'num_actions': 0,
                'child_ids': [-1, -1, -1]
            })
            return

        round_over = False
        if len(actions) >= 2:
            if actions[-2:] == ['c', 'c']:
                round_over = True
            elif actions[-1] == 'c' and actions[-2] == 'b':
                round_over = True

        if round_over:
            node_type = NODE_TO_R2 if round_num == 1 else NODE_TERMINAL_SHOWDOWN
            nodes.append({
                'type': node_type,
                'action_id': action_id,
                'p0': p0, 'p1': p1,
                'fold_player': 0, 'player': 0, 'num_actions': 0,
                'child_ids': [-1, -1, -1]
            })
            return

        facing_bet = self._facing_bet(actions)
        if facing_bet:
            num_actions = 3 if num_bets < MAX_RAISES else 2
        else:
            num_actions = 2 if num_bets < MAX_RAISES else 1

        if facing_bet:
            children = [actions + ['f'], actions + ['c']]
            if num_actions == 3:
                children.append(actions + ['b'])
        else:
            children = [actions + ['c']]
            if num_actions == 2:
                children.append(actions + ['b'])

        child_ids = [self._get_action_id(tuple(c)) for c in children]
        while len(child_ids) < 3:
            child_ids.append(-1)

        nodes.append({
            'type': NODE_DECISION,
            'action_id': action_id,
            'player': player,
            'num_actions': num_actions,
            'p0': p0, 'p1': p1,
            'fold_player': 0,
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

    def _get_action_id(self, actions):
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

    def _convert_r1(self, nodes):
        n = len(nodes)
        self.r1_num_nodes = n
        self.r1_type = np.array([nd['type'] for nd in nodes], dtype=np.int32)
        self.r1_action_id = np.array([nd['action_id'] for nd in nodes], dtype=np.int32)
        self.r1_player = np.array([nd['player'] for nd in nodes], dtype=np.int32)
        self.r1_num_actions = np.array([nd['num_actions'] for nd in nodes], dtype=np.int32)
        self.r1_p0 = np.array([nd['p0'] for nd in nodes], dtype=np.float32)
        self.r1_p1 = np.array([nd['p1'] for nd in nodes], dtype=np.float32)
        self.r1_fold_player = np.array([nd['fold_player'] for nd in nodes], dtype=np.int32)
        self.r1_child_ids = np.array([nd['child_ids'] for nd in nodes], dtype=np.int32)

        # Build aid -> node_idx mapping
        max_aid = max(self.r1_action_id) + 1
        self.r1_aid_to_idx = np.full(max_aid, -1, dtype=np.int32)
        for i, aid in enumerate(self.r1_action_id):
            self.r1_aid_to_idx[aid] = i

    def _convert_r2(self, nodes):
        n = len(nodes)
        self.r2_num_nodes = n
        self.r2_type = np.array([nd['type'] for nd in nodes], dtype=np.int32)
        self.r2_action_id = np.array([nd['action_id'] for nd in nodes], dtype=np.int32)
        self.r2_player = np.array([nd['player'] for nd in nodes], dtype=np.int32)
        self.r2_num_actions = np.array([nd['num_actions'] for nd in nodes], dtype=np.int32)
        self.r2_p0 = np.array([nd['p0'] for nd in nodes], dtype=np.float32)
        self.r2_p1 = np.array([nd['p1'] for nd in nodes], dtype=np.float32)
        self.r2_fold_player = np.array([nd['fold_player'] for nd in nodes], dtype=np.int32)
        self.r2_child_ids = np.array([nd['child_ids'] for nd in nodes], dtype=np.int32)

        max_aid = max(self.r2_action_id) + 1
        self.r2_aid_to_idx = np.full(max_aid, -1, dtype=np.int32)
        for i, aid in enumerate(self.r2_action_id):
            self.r2_aid_to_idx[aid] = i

    def _build_infosets(self, r1_nodes, r2_nodes):
        self.infoset_key_to_idx = {}
        idx = 0

        # R1 infosets - (node_idx, rank) -> h_idx
        self.r1_infoset_keys = np.full((len(r1_nodes), 3), -1, dtype=np.int32)

        for i, nd in enumerate(r1_nodes):
            if nd['type'] == NODE_DECISION:
                aid = nd['action_id']
                for rank in range(3):
                    key = (1, rank, -1, aid)
                    if key not in self.infoset_key_to_idx:
                        self.infoset_key_to_idx[key] = idx
                        self.r1_infoset_keys[i, rank] = idx
                        idx += 1
                    else:
                        self.r1_infoset_keys[i, rank] = self.infoset_key_to_idx[key]

        # R2 infosets - (node_idx, rank, comm) -> h_idx
        self.r2_infoset_keys = np.full((len(r2_nodes), 3, 3), -1, dtype=np.int32)

        for i, nd in enumerate(r2_nodes):
            if nd['type'] == NODE_DECISION:
                aid = nd['action_id']
                for rank in range(3):
                    for comm in range(3):
                        key = (2, rank, comm, aid)
                        if key not in self.infoset_key_to_idx:
                            self.infoset_key_to_idx[key] = idx
                            self.r2_infoset_keys[i, rank, comm] = idx
                            idx += 1
                        else:
                            self.r2_infoset_keys[i, rank, comm] = self.infoset_key_to_idx[key]

        self.num_infosets = idx
        self.infoset_num_actions = np.zeros(idx, dtype=np.int32)

        for key, h_idx in self.infoset_key_to_idx.items():
            round_num, rank, comm, aid = key
            if round_num == 1:
                node_idx = self.r1_aid_to_idx[aid]
                self.infoset_num_actions[h_idx] = self.r1_num_actions[node_idx]
            else:
                node_idx = self.r2_aid_to_idx[aid]
                self.infoset_num_actions[h_idx] = self.r2_num_actions[node_idx]

    def _get_strategy_all(self):
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
        regret_updates = np.zeros_like(self._cumulative_regret)

        for scenario in self.scenarios:
            deal_idx = int(scenario[0])
            hero_rank = int(scenario[1])
            villain_rank = int(scenario[2])
            comm_rank = int(scenario[3])
            hero_str = int(scenario[4])
            villain_str = int(scenario[5])

            for player in range(2):
                player_rank = hero_rank if player == 0 else villain_rank

                process_round1_nodes(
                    self.r1_num_nodes,
                    self.r1_type, self.r1_action_id, self.r1_player,
                    self.r1_num_actions, self.r1_p0, self.r1_p1,
                    self.r1_fold_player, self.r1_child_ids, self.r1_aid_to_idx,
                    self.r1_infoset_keys,
                    self.r2_num_nodes,
                    self.r2_type, self.r2_action_id, self.r2_player,
                    self.r2_num_actions, self.r2_p0, self.r2_p1,
                    self.r2_fold_player, self.r2_child_ids, self.r2_aid_to_idx,
                    self.r2_infoset_keys,
                    hero_rank, villain_rank, comm_rank, hero_str, villain_str,
                    strat, deal_idx, player, player_rank, weight,
                    regret_updates
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

    def exploitability(self) -> float:
        """Compute exploitability using vanilla CFR."""
        from gpu_poker_cfr.solvers.vanilla import VanillaCFR

        vanilla = VanillaCFR(self.game, backend='numpy')
        avg_strat = self.average_strategy

        vanilla_strat = np.zeros(vanilla.matrices.num_infoset_actions, dtype=np.float32)

        # Map our strategy to vanilla format
        for h_idx in range(self.num_infosets):
            start = vanilla.matrices.infoset_action_offsets[h_idx]
            end = vanilla.matrices.infoset_action_offsets[h_idx + 1]
            num_actions = end - start

            # Average over all deals
            avg = avg_strat[h_idx, :num_actions, :].mean(axis=1)
            vanilla_strat[start:end] = avg

        vanilla._cumulative_strategy = vanilla.backend.dense_to_backend(vanilla_strat)
        vanilla.iterations = 1

        return vanilla.exploitability()
