"""
Highly Optimized Semi-Vector MCCFR for Leduc Poker.

Key optimizations:
1. Fully vectorize across all 30 deals simultaneously
2. Pre-compute all tree sequences as numpy arrays
3. Batch strategy lookups and regret updates
4. Minimize Python loops

Target: 20-50x speedup over naive implementation.
"""

import numpy as np
from typing import List, Tuple

from gpu_poker_cfr.games.base import Game
from gpu_poker_cfr.games.leduc import (
    LeducPoker, JACK, QUEEN, KING,
    ANTE, ROUND1_BET, ROUND2_BET, MAX_RAISES
)


# Constants
LEDUC_NUM_RANKS = 3
LEDUC_NUM_CARDS = 6
LEDUC_NUM_DEALS = 30


def card_to_rank(card: int) -> int:
    return card // 2


def get_all_deals() -> np.ndarray:
    """Return (30, 2) array of [hero_card, villain_card]."""
    deals = []
    for h in range(6):
        for v in range(6):
            if h != v:
                deals.append([h, v])
    return np.array(deals, dtype=np.int32)


class SemiVectorLeducV2:
    """
    Highly optimized Semi-Vector MCCFR for Leduc.

    Vectorizes ALL operations across the 30 private deals.
    """

    def __init__(self, game: Game, sample_boards: bool = False, num_samples: int = 1):
        if not isinstance(game, LeducPoker):
            raise ValueError("Only LeducPoker supported")

        self.game = game
        self.sample_boards = sample_boards
        self.num_samples = num_samples

        # Pre-compute deals
        self._deals = get_all_deals()  # (30, 2)
        self._deal_ranks = self._deals // 2  # (30, 2) - hero/villain ranks

        # Build action sequences
        self._build_sequences()

        # Regrets: (num_infosets, max_actions, 30 deals)
        self._cumulative_regret = np.zeros(
            (self.num_infosets, 3, LEDUC_NUM_DEALS), dtype=np.float32
        )
        self._cumulative_strategy = np.zeros(
            (self.num_infosets, 3, LEDUC_NUM_DEALS), dtype=np.float32
        )

        self.iterations = 0

    def _build_sequences(self):
        """Build all betting sequences as arrays."""
        # Round 1 sequences
        r1_seqs = []
        self._enumerate_seqs(r1_seqs, [], 0, ANTE, ANTE, 0, round_num=1)

        # Round 2 sequences
        r2_seqs = []
        self._enumerate_seqs(r2_seqs, [], 0, 0, 0, 0, round_num=2)

        self.r1_seqs = r1_seqs
        self.r2_seqs = r2_seqs

        # Build infoset index
        self._build_infosets()

    def _enumerate_seqs(self, seqs, actions, player, p0, p1, num_bets, round_num):
        """Enumerate betting sequences."""
        bet_size = ROUND1_BET if round_num == 1 else ROUND2_BET

        seq = {
            'actions': tuple(actions),
            'player': player,
            'p0': p0,
            'p1': p1,
            'terminal': False,
            'showdown': False,
            'fold_player': -1,
            'to_next': False,
        }

        # Check terminal conditions
        if actions and actions[-1] == 'f':
            seq['terminal'] = True
            seq['fold_player'] = len(actions) % 2  # Who took fold action
            seqs.append(seq)
            return

        if len(actions) >= 2:
            if actions[-2:] == ['c', 'c']:
                seq['terminal'] = round_num == 2
                seq['showdown'] = round_num == 2
                seq['to_next'] = round_num == 1
                seqs.append(seq)
                return
            if actions[-1] == 'c' and len(actions) >= 2 and actions[-2] == 'b':
                seq['terminal'] = round_num == 2
                seq['showdown'] = round_num == 2
                seq['to_next'] = round_num == 1
                seqs.append(seq)
                return

        seqs.append(seq)

        # Generate children
        facing_bet = self._is_facing_bet(actions)
        next_player = 1 - player

        if facing_bet:
            # Fold
            self._enumerate_seqs(seqs, actions + ['f'], next_player, p0, p1, num_bets, round_num)
            # Call
            new_p0 = max(p0, p1) if player == 0 else p0
            new_p1 = max(p0, p1) if player == 1 else p1
            self._enumerate_seqs(seqs, actions + ['c'], next_player, new_p0, new_p1, num_bets, round_num)
            # Raise
            if num_bets < MAX_RAISES:
                if player == 0:
                    new_p0 = max(p0, p1) + bet_size
                else:
                    new_p1 = max(p0, p1) + bet_size
                self._enumerate_seqs(seqs, actions + ['b'], next_player,
                                    new_p0 if player == 0 else p0,
                                    new_p1 if player == 1 else p1,
                                    num_bets + 1, round_num)
        else:
            # Check
            self._enumerate_seqs(seqs, actions + ['c'], next_player, p0, p1, num_bets, round_num)
            # Bet
            if num_bets < MAX_RAISES:
                if player == 0:
                    new_p0 = p0 + bet_size
                else:
                    new_p1 = p1 + bet_size
                self._enumerate_seqs(seqs, actions + ['b'], next_player,
                                    new_p0 if player == 0 else p0,
                                    new_p1 if player == 1 else p1,
                                    num_bets + 1, round_num)

    def _is_facing_bet(self, actions):
        if not actions:
            return False
        for i in range(len(actions) - 1, -1, -1):
            if actions[i] == 'b':
                return True
            if actions[i] == 'c' and i > 0 and actions[i-1] == 'b':
                return False
        return False

    def _build_infosets(self):
        """Build infoset mapping."""
        self.infoset_map = {}
        self.infoset_info = []
        idx = 0

        # Round 1: (rank, actions) - no community
        for seq in self.r1_seqs:
            if not seq['terminal'] and not seq['to_next']:
                for rank in range(3):
                    key = (1, rank, -1, seq['actions'])
                    if key not in self.infoset_map:
                        self.infoset_map[key] = idx
                        self.infoset_info.append({
                            'round': 1, 'rank': rank, 'comm': -1,
                            'actions': seq['actions'], 'player': seq['player']
                        })
                        idx += 1

        # Round 2: (rank, community, actions)
        for seq in self.r2_seqs:
            if not seq['terminal']:
                for rank in range(3):
                    for comm in range(3):
                        key = (2, rank, comm, seq['actions'])
                        if key not in self.infoset_map:
                            self.infoset_map[key] = idx
                            self.infoset_info.append({
                                'round': 2, 'rank': rank, 'comm': comm,
                                'actions': seq['actions'], 'player': seq['player']
                            })
                            idx += 1

        self.num_infosets = idx

        # Build arrays
        self.infoset_player = np.array([i['player'] for i in self.infoset_info], dtype=np.int32)
        self.infoset_rank = np.array([i['rank'] for i in self.infoset_info], dtype=np.int32)
        self.infoset_num_actions = np.zeros(self.num_infosets, dtype=np.int32)

        for h_idx, info in enumerate(self.infoset_info):
            actions = info['actions']
            facing = self._is_facing_bet(list(actions))
            num_bets = sum(1 for a in actions if a == 'b')
            if facing:
                self.infoset_num_actions[h_idx] = 3 if num_bets < MAX_RAISES else 2
            else:
                self.infoset_num_actions[h_idx] = 2 if num_bets < MAX_RAISES else 1

    def _get_strategy(self) -> np.ndarray:
        """Get strategy array (num_infosets, 3, 30)."""
        pos_reg = np.maximum(self._cumulative_regret, 0)
        reg_sum = pos_reg.sum(axis=1, keepdims=True)

        # Uniform fallback
        uniform = np.zeros_like(self._cumulative_regret)
        for h in range(self.num_infosets):
            n = self.infoset_num_actions[h]
            uniform[h, :n, :] = 1.0 / n

        safe_sum = np.where(reg_sum > 0, reg_sum, 1.0)
        strat = np.where(reg_sum > 0, pos_reg / safe_sum, uniform)

        # Zero invalid actions
        for h in range(self.num_infosets):
            n = self.infoset_num_actions[h]
            strat[h, n:, :] = 0

        return strat

    def iterate(self, n: int = 1):
        for _ in range(n):
            self._iterate_once()
            self.iterations += 1

    def _iterate_once(self):
        """Single iteration with vectorized deal processing."""
        t = self.iterations + 1
        strat = self._get_strategy()

        # For each board card (or sample)
        boards_per_deal = []
        for deal_idx in range(LEDUC_NUM_DEALS):
            h, v = self._deals[deal_idx]
            remaining = [c for c in range(6) if c != h and c != v]
            if self.sample_boards:
                boards = list(np.random.choice(remaining, min(self.num_samples, len(remaining)), replace=False))
            else:
                boards = remaining
            boards_per_deal.append(boards)

        # Process all deals for each board configuration
        # Group deals by number of boards (all have 4 in full enum)
        for board_idx in range(4 if not self.sample_boards else self.num_samples):
            # Get board card for each deal
            board_cards = []
            valid_deals = []
            for deal_idx in range(LEDUC_NUM_DEALS):
                if board_idx < len(boards_per_deal[deal_idx]):
                    board_cards.append(boards_per_deal[deal_idx][board_idx])
                    valid_deals.append(deal_idx)

            if not valid_deals:
                continue

            board_cards = np.array(board_cards, dtype=np.int32)
            valid_deals = np.array(valid_deals, dtype=np.int32)
            board_ranks = board_cards // 2

            # Vectorized update for this board configuration
            self._update_vectorized(strat, valid_deals, board_ranks, t)

    def _update_vectorized(self, strat, deal_indices, board_ranks, t):
        """Update regrets for multiple deals at once."""
        n_deals = len(deal_indices)
        weight = 1.0 / LEDUC_NUM_DEALS / 4  # Assuming 4 boards

        # Get ranks for these deals
        hero_ranks = self._deal_ranks[deal_indices, 0]
        villain_ranks = self._deal_ranks[deal_indices, 1]

        # Compute hand strengths
        hero_pair = (hero_ranks == board_ranks).astype(np.int32)
        villain_pair = (villain_ranks == board_ranks).astype(np.int32)
        hero_strength = hero_pair * 100 + hero_ranks
        villain_strength = villain_pair * 100 + villain_ranks

        # Process round 1
        r1_values = {}  # action_tuple -> (n_deals,) array of values

        for seq in reversed(self.r1_seqs):
            actions = seq['actions']

            if seq['terminal']:
                vals = np.zeros((n_deals, 2), dtype=np.float32)
                fp = seq['fold_player']
                if fp == 0:
                    vals[:, 0] = -seq['p0']
                    vals[:, 1] = seq['p0']
                else:
                    vals[:, 0] = seq['p1']
                    vals[:, 1] = -seq['p1']
                r1_values[actions] = vals

            elif seq['to_next']:
                # Compute round 2 EV
                r2_ev = self._compute_r2_vectorized(
                    strat, deal_indices, hero_ranks, villain_ranks,
                    board_ranks, hero_strength, villain_strength,
                    seq['p0'], seq['p1']
                )
                r1_values[actions] = r2_ev

            else:
                # Decision node
                player = seq['player']
                acting_ranks = hero_ranks if player == 0 else villain_ranks

                facing = self._is_facing_bet(list(actions))
                num_bets = sum(1 for a in actions if a == 'b')

                if facing:
                    children = [('f',), ('c',)]
                    if num_bets < MAX_RAISES:
                        children.append(('b',))
                else:
                    children = [('c',)]
                    if num_bets < MAX_RAISES:
                        children.append(('b',))

                # Get child values
                child_vals = []
                for c in children:
                    child_key = actions + c
                    if child_key in r1_values:
                        child_vals.append(r1_values[child_key])
                    else:
                        child_vals.append(np.zeros((n_deals, 2), dtype=np.float32))

                # Compute weighted EV for each deal
                ev = np.zeros((n_deals, 2), dtype=np.float32)

                for rank in range(3):
                    rank_mask = acting_ranks == rank
                    if not np.any(rank_mask):
                        continue

                    key = (1, rank, -1, actions)
                    if key not in self.infoset_map:
                        continue

                    h_idx = self.infoset_map[key]
                    n_act = len(children)

                    # Get strategy for deals with this rank
                    deal_mask = np.where(rank_mask)[0]
                    global_deal_idx = deal_indices[deal_mask]

                    s = strat[h_idx, :n_act, global_deal_idx]  # (n_act, n_matching)

                    for a_idx in range(n_act):
                        # s[a_idx, :] is (n_matching,), child_vals is (n_matching, 2)
                        ev[deal_mask] += s[a_idx, :][:, np.newaxis] * child_vals[a_idx][deal_mask]

                    # Update regrets for player's infosets
                    for p in range(2):
                        if player != p:
                            continue
                        p_ranks = hero_ranks if p == 0 else villain_ranks
                        p_mask = (p_ranks == rank) & rank_mask

                        if not np.any(p_mask):
                            continue

                        matching_idx = np.where(p_mask)[0]
                        global_idx = deal_indices[matching_idx]

                        for a_idx in range(n_act):
                            av = child_vals[a_idx][matching_idx, p]
                            ev_p = ev[matching_idx, p]
                            instant_regret = av - ev_p
                            self._cumulative_regret[h_idx, a_idx, global_idx] += weight * instant_regret

                r1_values[actions] = ev

        # Update cumulative strategy
        self._cumulative_strategy += t * strat

    def _compute_r2_vectorized(self, strat, deal_indices, hero_ranks, villain_ranks,
                               board_ranks, hero_strength, villain_strength, p0_base, p1_base):
        """Compute round 2 EV for all deals."""
        n_deals = len(deal_indices)
        r2_vals = {}

        for seq in reversed(self.r2_seqs):
            actions = seq['actions']
            p0 = p0_base + seq['p0']
            p1 = p1_base + seq['p1']

            if seq['terminal']:
                vals = np.zeros((n_deals, 2), dtype=np.float32)
                if seq['showdown']:
                    hero_wins = hero_strength > villain_strength
                    villain_wins = villain_strength > hero_strength
                    vals[hero_wins, 0] = p1
                    vals[hero_wins, 1] = -p1
                    vals[villain_wins, 0] = -p0
                    vals[villain_wins, 1] = p0
                else:
                    fp = seq['fold_player']
                    if fp == 0:
                        vals[:, 0] = -p0
                        vals[:, 1] = p0
                    else:
                        vals[:, 0] = p1
                        vals[:, 1] = -p1
                r2_vals[actions] = vals

            else:
                player = seq['player']
                acting_ranks = hero_ranks if player == 0 else villain_ranks

                facing = self._is_facing_bet(list(actions))
                num_bets = sum(1 for a in actions if a == 'b')

                if facing:
                    children = [('f',), ('c',)]
                    if num_bets < MAX_RAISES:
                        children.append(('b',))
                else:
                    children = [('c',)]
                    if num_bets < MAX_RAISES:
                        children.append(('b',))

                child_vals = []
                for c in children:
                    child_key = actions + c
                    if child_key in r2_vals:
                        child_vals.append(r2_vals[child_key])
                    else:
                        child_vals.append(np.zeros((n_deals, 2), dtype=np.float32))

                ev = np.zeros((n_deals, 2), dtype=np.float32)

                for rank in range(3):
                    for comm in range(3):
                        mask = (acting_ranks == rank) & (board_ranks == comm)
                        if not np.any(mask):
                            continue

                        key = (2, rank, comm, actions)
                        if key not in self.infoset_map:
                            continue

                        h_idx = self.infoset_map[key]
                        n_act = len(children)

                        deal_mask = np.where(mask)[0]
                        global_idx = deal_indices[deal_mask]

                        s = strat[h_idx, :n_act, global_idx]  # (n_act, n_matching)

                        for a_idx in range(n_act):
                            # s[a_idx, :] is (n_matching,), child_vals is (n_matching, 2)
                            ev[deal_mask] += s[a_idx, :][:, np.newaxis] * child_vals[a_idx][deal_mask]

                r2_vals[actions] = ev

        return r2_vals.get((), np.zeros((n_deals, 2), dtype=np.float32))

    def solve(self, iterations: int = 1000):
        self.iterate(iterations)

    @property
    def average_strategy(self):
        total = self._cumulative_strategy.sum(axis=1, keepdims=True)
        safe = np.where(total > 0, total, 1.0)
        return np.where(total > 0, self._cumulative_strategy / safe, 1.0 / 3)
