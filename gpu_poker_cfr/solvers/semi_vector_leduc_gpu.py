"""
GPU-Accelerated Semi-Vector MCCFR for Leduc Poker using CuPy.

Based on semi_vector_leduc_opt.py with GPU acceleration:
1. Regret/strategy arrays on GPU
2. Vectorized strategy computation on GPU
3. Batch regret updates across all deals simultaneously
"""

import numpy as np
import cupy as cp
from typing import List, Tuple, Dict

from gpu_poker_cfr.games.base import Game
from gpu_poker_cfr.games.leduc import (
    LeducPoker, ANTE, ROUND1_BET, ROUND2_BET, MAX_RAISES
)


LEDUC_NUM_CARDS = 6
LEDUC_NUM_DEALS = 30
LEDUC_NUM_RANKS = 3


def card_to_rank(c: int) -> int:
    return c // 2


def get_all_deals():
    return [(h, v) for h in range(6) for v in range(6) if h != v]


class SemiVectorLeducGPU:
    """GPU-Accelerated Semi-Vector MCCFR for Leduc."""

    def __init__(self, game: Game, sample_boards: bool = False, num_samples: int = 1):
        if not isinstance(game, LeducPoker):
            raise ValueError("Only LeducPoker supported")

        self.game = game
        self.sample_boards = sample_boards
        self.num_samples = num_samples

        # Build tree structure (CPU)
        self._build_tree()

        # Regrets on GPU: (num_infosets, max_actions, num_deals)
        self._cumulative_regret = cp.zeros(
            (self.num_infosets, 3, LEDUC_NUM_DEALS), dtype=cp.float32
        )
        self._cumulative_strategy = cp.zeros(
            (self.num_infosets, 3, LEDUC_NUM_DEALS), dtype=cp.float32
        )

        # Pre-compute uniform strategy on GPU
        self._uniform = cp.zeros((self.num_infosets, 3, LEDUC_NUM_DEALS), dtype=cp.float32)
        for h in range(self.num_infosets):
            n = self.infoset_num_actions[h]
            self._uniform[h, :n, :] = 1.0 / n

        # Pre-compute deal info for vectorization
        self._precompute_deal_info()

        self.iterations = 0

    def _precompute_deal_info(self):
        """Pre-compute deal-related arrays for vectorization."""
        all_deals = get_all_deals()

        # Hero/villain ranks for each deal
        self.deal_hero_rank = np.array([card_to_rank(h) for h, v in all_deals], dtype=np.int32)
        self.deal_villain_rank = np.array([card_to_rank(v) for h, v in all_deals], dtype=np.int32)

        # Remaining cards for each deal
        self.deal_remaining = []
        for h, v in all_deals:
            self.deal_remaining.append([c for c in range(6) if c != h and c != v])

    def _build_tree(self):
        """Build pre-computed tree structure."""
        self.r1_nodes = []
        self.r2_nodes = []
        self._build_round_tree(self.r1_nodes, 1, [], 0, ANTE, ANTE, 0)
        self._build_round_tree(self.r2_nodes, 2, [], 0, 0, 0, 0)

        # Build infoset index
        self.infoset_key_to_idx = {}
        self.infoset_info = []
        idx = 0

        # Round 1 infosets
        for node in self.r1_nodes:
            if node['type'] == 'decision':
                for rank in range(3):
                    key = (1, rank, -1, node['action_id'])
                    if key not in self.infoset_key_to_idx:
                        self.infoset_key_to_idx[key] = idx
                        self.infoset_info.append({
                            'player': node['player'],
                            'rank': rank,
                            'num_actions': node['num_actions']
                        })
                        idx += 1

        # Round 2 infosets
        for node in self.r2_nodes:
            if node['type'] == 'decision':
                for rank in range(3):
                    for comm in range(3):
                        key = (2, rank, comm, node['action_id'])
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
        self.infoset_player = np.array(
            [i['player'] for i in self.infoset_info], dtype=np.int32
        )
        self.infoset_rank = np.array(
            [i['rank'] for i in self.infoset_info], dtype=np.int32
        )

        self._action_id_counter = 0

    def _get_action_id(self, actions: tuple) -> int:
        """Get unique integer ID for action sequence."""
        if not hasattr(self, '_action_map'):
            self._action_map = {}
        if actions not in self._action_map:
            self._action_map[actions] = len(self._action_map)
        return self._action_map[actions]

    def _build_round_tree(self, nodes: list, round_num: int, actions: list,
                          player: int, p0: int, p1: int, num_bets: int):
        """Build tree for a single round."""
        bet_size = ROUND1_BET if round_num == 1 else ROUND2_BET
        action_tuple = tuple(actions)
        action_id = self._get_action_id(action_tuple)

        if actions and actions[-1] == 'f':
            fold_player = (len(actions) - 1) % 2
            nodes.append({
                'type': 'terminal',
                'actions': action_tuple,
                'action_id': action_id,
                'fold': True,
                'fold_player': fold_player,
                'p0': p0, 'p1': p1
            })
            return

        round_over = False
        if len(actions) >= 2:
            if actions[-2:] == ['c', 'c']:
                round_over = True
            elif actions[-1] == 'c' and len(actions) >= 2 and actions[-2] == 'b':
                round_over = True

        if round_over:
            nodes.append({
                'type': 'terminal' if round_num == 2 else 'to_r2',
                'actions': action_tuple,
                'action_id': action_id,
                'fold': False,
                'showdown': round_num == 2,
                'p0': p0, 'p1': p1
            })
            return

        facing_bet = self._facing_bet(actions)
        if facing_bet:
            num_actions = 3 if num_bets < MAX_RAISES else 2
        else:
            num_actions = 2 if num_bets < MAX_RAISES else 1

        nodes.append({
            'type': 'decision',
            'actions': action_tuple,
            'action_id': action_id,
            'player': player,
            'facing_bet': facing_bet,
            'num_actions': num_actions,
            'p0': p0, 'p1': p1
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

    def _facing_bet(self, actions):
        if not actions:
            return False
        for i in range(len(actions) - 1, -1, -1):
            if actions[i] == 'b':
                return True
            if actions[i] == 'c' and i > 0 and actions[i-1] == 'b':
                return False
        return False

    def _get_strategy_all_gpu(self) -> cp.ndarray:
        """Get strategy for all infosets (GPU vectorized)."""
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
        for _ in range(n):
            self._single_iteration()
            self.iterations += 1

    def _single_iteration(self):
        t = self.iterations + 1

        # Get strategy on GPU
        strat_gpu = self._get_strategy_all_gpu()
        # Transfer to CPU for tree traversal
        strat = cp.asnumpy(strat_gpu)

        all_deals = get_all_deals()
        deal_weight = 1.0 / len(all_deals)

        # Accumulate regret updates on CPU, then batch update GPU
        regret_updates = np.zeros((self.num_infosets, 3, LEDUC_NUM_DEALS), dtype=np.float32)

        for deal_idx, (hero, villain) in enumerate(all_deals):
            remaining = self.deal_remaining[deal_idx]

            if self.sample_boards:
                boards = list(np.random.choice(remaining, min(self.num_samples, 4), replace=False))
                board_weight = 1.0
            else:
                boards = remaining
                board_weight = 1.0 / 4

            for comm in boards:
                weight = deal_weight * board_weight

                for player in range(2):
                    self._update_regrets_cpu(
                        hero, villain, comm, player,
                        strat, deal_idx, weight, regret_updates
                    )

        # Batch update GPU arrays
        self._cumulative_regret += cp.asarray(regret_updates)
        self._cumulative_strategy += t * strat_gpu

    def _update_regrets_cpu(self, hero: int, villain: int, comm: int,
                            player: int, strat: np.ndarray, deal_idx: int,
                            weight: float, regret_updates: np.ndarray):
        """Update regrets using iterative tree traversal (CPU)."""
        hero_rank = card_to_rank(hero)
        villain_rank = card_to_rank(villain)
        comm_rank = card_to_rank(comm)

        player_rank = hero_rank if player == 0 else villain_rank

        hero_pair = 1 if hero_rank == comm_rank else 0
        villain_pair = 1 if villain_rank == comm_rank else 0
        hero_strength = hero_pair * 100 + hero_rank
        villain_strength = villain_pair * 100 + villain_rank

        r1_values = {}

        for node in reversed(self.r1_nodes):
            aid = node['action_id']

            if node['type'] == 'terminal':
                if node['fold']:
                    fp = node['fold_player']
                    if fp == 0:
                        r1_values[aid] = (-node['p0'], node['p0'])
                    else:
                        r1_values[aid] = (node['p1'], -node['p1'])
                else:
                    r1_values[aid] = (0.0, 0.0)

            elif node['type'] == 'to_r2':
                r2_ev = self._compute_r2_cpu(
                    hero_rank, villain_rank, comm_rank,
                    hero_strength, villain_strength,
                    node['p0'], node['p1'],
                    strat, deal_idx, regret_updates
                )
                r1_values[aid] = r2_ev

            else:
                acting_player = node['player']
                acting_rank = hero_rank if acting_player == 0 else villain_rank
                n_actions = node['num_actions']

                actions = node['actions']
                if node['facing_bet']:
                    children = [actions + ('f',), actions + ('c',)]
                    if n_actions == 3:
                        children.append(actions + ('b',))
                else:
                    children = [actions + ('c',)]
                    if n_actions == 2:
                        children.append(actions + ('b',))

                child_values = []
                for c in children:
                    cid = self._get_action_id(c)
                    if cid in r1_values:
                        child_values.append(r1_values[cid])
                    else:
                        child_values.append((0.0, 0.0))

                key = (1, acting_rank, -1, aid)
                if key in self.infoset_key_to_idx:
                    h_idx = self.infoset_key_to_idx[key]
                    s = strat[h_idx, :n_actions, deal_idx]

                    ev0 = sum(s[i] * child_values[i][0] for i in range(n_actions))
                    ev1 = sum(s[i] * child_values[i][1] for i in range(n_actions))
                    r1_values[aid] = (ev0, ev1)

                    if acting_player == player and acting_rank == player_rank:
                        for a_idx in range(n_actions):
                            av = child_values[a_idx][player]
                            ev = r1_values[aid][player]
                            regret_updates[h_idx, a_idx, deal_idx] += weight * (av - ev)
                else:
                    r1_values[aid] = (0.0, 0.0)

    def _compute_r2_cpu(self, hero_rank, villain_rank, comm_rank,
                        hero_strength, villain_strength, p0_base, p1_base,
                        strat, deal_idx, regret_updates):
        """Compute round 2 EV (CPU)."""
        player_rank = [hero_rank, villain_rank]

        r2_values = {}

        for node in reversed(self.r2_nodes):
            aid = node['action_id']
            p0 = p0_base + node['p0']
            p1 = p1_base + node['p1']

            if node['type'] == 'terminal':
                if node.get('fold', False):
                    fp = node['fold_player']
                    if fp == 0:
                        r2_values[aid] = (-p0, p0)
                    else:
                        r2_values[aid] = (p1, -p1)
                elif node.get('showdown', False):
                    if hero_strength > villain_strength:
                        r2_values[aid] = (p1, -p1)
                    elif villain_strength > hero_strength:
                        r2_values[aid] = (-p0, p0)
                    else:
                        r2_values[aid] = (0.0, 0.0)
                else:
                    r2_values[aid] = (0.0, 0.0)

            else:
                acting_player = node['player']
                acting_rank = player_rank[acting_player]
                n_actions = node['num_actions']

                actions = node['actions']
                if node['facing_bet']:
                    children = [actions + ('f',), actions + ('c',)]
                    if n_actions == 3:
                        children.append(actions + ('b',))
                else:
                    children = [actions + ('c',)]
                    if n_actions == 2:
                        children.append(actions + ('b',))

                child_values = []
                for c in children:
                    cid = self._get_action_id(c)
                    if cid in r2_values:
                        child_values.append(r2_values[cid])
                    else:
                        child_values.append((0.0, 0.0))

                key = (2, acting_rank, comm_rank, aid)
                if key in self.infoset_key_to_idx:
                    h_idx = self.infoset_key_to_idx[key]
                    s = strat[h_idx, :n_actions, deal_idx]

                    ev0 = sum(s[i] * child_values[i][0] for i in range(n_actions))
                    ev1 = sum(s[i] * child_values[i][1] for i in range(n_actions))
                    r2_values[aid] = (ev0, ev1)

                    for p in range(2):
                        if acting_player == p and acting_rank == player_rank[p]:
                            for a_idx in range(n_actions):
                                av = child_values[a_idx][p]
                                ev = r2_values[aid][p]
                                regret_updates[h_idx, a_idx, deal_idx] += (1.0/30/4) * (av - ev)
                else:
                    r2_values[aid] = (0.0, 0.0)

        root_id = self._get_action_id(())
        return r2_values.get(root_id, (0.0, 0.0))

    def solve(self, iterations: int = 1000):
        self.iterate(iterations)

    @property
    def average_strategy(self):
        total = self._cumulative_strategy.sum(axis=1, keepdims=True)
        safe = cp.where(total > 0, total, 1.0)
        return cp.where(total > 0, self._cumulative_strategy / safe, 1.0/3)

    def get_average_strategy_cpu(self):
        """Return average strategy as numpy array."""
        return cp.asnumpy(self.average_strategy)
