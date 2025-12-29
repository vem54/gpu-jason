"""
Fast Semi-Vector MCCFR for Leduc Poker.

Optimized version that:
1. Pre-computes game tree structure
2. Vectorizes across all 30 private deals
3. Uses numpy arrays instead of Python loops
4. Eliminates recursive function calls

Target: 10-50x speedup over naive implementation.
"""

import numpy as np
from typing import Literal, Optional, Tuple, List, Dict
from dataclasses import dataclass

from gpu_poker_cfr.games.base import Game, Player
from gpu_poker_cfr.games.leduc import (
    LeducPoker, JACK, QUEEN, KING, CARD_NAMES,
    ANTE, ROUND1_BET, ROUND2_BET, MAX_RAISES
)


# =============================================================================
# Constants
# =============================================================================

LEDUC_NUM_RANKS = 3
LEDUC_NUM_SUITS = 2
LEDUC_NUM_CARDS = 6
LEDUC_NUM_PRIVATE_DEALS = 30  # 6 * 5


def card_to_rank(card_idx: int) -> int:
    return card_idx // LEDUC_NUM_SUITS


def get_all_private_deals() -> List[Tuple[int, int]]:
    deals = []
    for hero in range(LEDUC_NUM_CARDS):
        for villain in range(LEDUC_NUM_CARDS):
            if hero != villain:
                deals.append((hero, villain))
    return deals


# =============================================================================
# Pre-computed Game Tree
# =============================================================================

@dataclass
class LeducTreeNode:
    """A node in the Leduc game tree."""
    id: int
    parent_id: int  # -1 for root
    player: int  # 0=P1, 1=P2, -1=terminal
    infoset_idx: int  # -1 for terminal/chance
    action_from_parent: int  # -1 for root
    is_terminal: bool
    round_num: int  # 1 or 2

    # For terminal nodes
    p0_contrib: int
    p1_contrib: int
    is_showdown: bool  # True if showdown, False if fold
    fold_player: int  # Which player folded (-1 if showdown)


class LeducGameTree:
    """
    Pre-computed Leduc game tree structure.

    The tree is independent of the actual cards dealt.
    We just need to know:
    - Tree topology (parent, children)
    - Which player acts at each node
    - Infoset assignments (based on private card rank and actions)
    - Terminal payoffs (as function of card comparison)
    """

    def __init__(self):
        self.nodes: List[LeducTreeNode] = []
        self.children: Dict[int, List[Tuple[int, int]]] = {}  # node_id -> [(action, child_id)]

        # Build separate trees for round 1 and round 2
        self._build_round1_tree()
        self._build_round2_tree()

        # Convert to numpy arrays for fast access
        self._to_numpy()

    def _build_round1_tree(self):
        """Build round 1 betting tree."""
        # Round 1: P1 acts first, can check/bet
        # Simpler approach: enumerate all action sequences

        self.round1_sequences = []
        self._enumerate_round1([], 0, ANTE, ANTE, 0)

    def _enumerate_round1(self, actions, player, p0_contrib, p1_contrib, num_bets):
        """Enumerate all round 1 action sequences."""
        seq = {
            'actions': tuple(actions),
            'player': player,
            'p0_contrib': p0_contrib,
            'p1_contrib': p1_contrib,
            'is_terminal': False,
            'to_round2': False,
        }

        # Check if round is over
        if len(actions) >= 2:
            last_two = actions[-2:]
            if last_two == ['c', 'c']:  # check-check
                seq['is_terminal'] = False
                seq['to_round2'] = True
                self.round1_sequences.append(seq)
                return
            if len(actions) >= 2 and actions[-1] == 'c':
                # Check if this is a call after bet
                bet_count = sum(1 for a in actions if a == 'b')
                call_count = sum(1 for a in actions if a == 'c')
                if bet_count > 0 and actions[-2] == 'b':
                    seq['is_terminal'] = False
                    seq['to_round2'] = True
                    self.round1_sequences.append(seq)
                    return

        if len(actions) >= 1 and actions[-1] == 'f':
            # Fold - terminal
            seq['is_terminal'] = True
            seq['fold_player'] = 1 - player  # Previous player folded
            self.round1_sequences.append(seq)
            return

        self.round1_sequences.append(seq)

        # Determine available actions
        facing_bet = self._facing_bet(actions)

        if facing_bet:
            # Can fold, call, or raise (if not max raises)
            # Fold
            self._enumerate_round1(actions + ['f'], 1 - player, p0_contrib, p1_contrib, num_bets)
            # Call
            new_p0 = max(p0_contrib, p1_contrib) if player == 0 else p0_contrib
            new_p1 = max(p0_contrib, p1_contrib) if player == 1 else p1_contrib
            self._enumerate_round1(actions + ['c'], 1 - player, new_p0, new_p1, num_bets)
            # Raise (if allowed)
            if num_bets < MAX_RAISES:
                if player == 0:
                    new_p0 = max(p0_contrib, p1_contrib) + ROUND1_BET
                else:
                    new_p1 = max(p0_contrib, p1_contrib) + ROUND1_BET
                self._enumerate_round1(actions + ['b'], 1 - player,
                                      new_p0 if player == 0 else p0_contrib,
                                      new_p1 if player == 1 else p1_contrib,
                                      num_bets + 1)
        else:
            # Can check or bet
            # Check
            self._enumerate_round1(actions + ['c'], 1 - player, p0_contrib, p1_contrib, num_bets)
            # Bet (if allowed)
            if num_bets < MAX_RAISES:
                if player == 0:
                    new_p0 = p0_contrib + ROUND1_BET
                else:
                    new_p1 = p1_contrib + ROUND1_BET
                self._enumerate_round1(actions + ['b'], 1 - player,
                                      new_p0 if player == 0 else p0_contrib,
                                      new_p1 if player == 1 else p1_contrib,
                                      num_bets + 1)

    def _facing_bet(self, actions):
        """Check if current player is facing a bet."""
        if not actions:
            return False
        # Find last bet and count calls after it
        for i in range(len(actions) - 1, -1, -1):
            if actions[i] == 'b':
                calls_after = sum(1 for a in actions[i+1:] if a == 'c')
                return calls_after == 0
        return False

    def _build_round2_tree(self):
        """Build round 2 betting tree (similar structure to round 1)."""
        self.round2_sequences = []
        self._enumerate_round2([], 0, 0, 0, 0)  # Contributions will be added from round 1

    def _enumerate_round2(self, actions, player, p0_contrib, p1_contrib, num_bets):
        """Enumerate all round 2 action sequences."""
        seq = {
            'actions': tuple(actions),
            'player': player,
            'p0_contrib_delta': p0_contrib,  # Delta from round 1
            'p1_contrib_delta': p1_contrib,
            'is_terminal': False,
            'is_showdown': False,
        }

        # Check if round is over
        if len(actions) >= 2:
            last_two = actions[-2:]
            if last_two == ['c', 'c']:  # check-check
                seq['is_terminal'] = True
                seq['is_showdown'] = True
                self.round2_sequences.append(seq)
                return
            if len(actions) >= 2 and actions[-1] == 'c':
                if actions[-2] == 'b':
                    seq['is_terminal'] = True
                    seq['is_showdown'] = True
                    self.round2_sequences.append(seq)
                    return

        if len(actions) >= 1 and actions[-1] == 'f':
            seq['is_terminal'] = True
            seq['fold_player'] = 1 - player
            self.round2_sequences.append(seq)
            return

        self.round2_sequences.append(seq)

        facing_bet = self._facing_bet(actions)

        if facing_bet:
            self._enumerate_round2(actions + ['f'], 1 - player, p0_contrib, p1_contrib, num_bets)
            new_p0 = max(p0_contrib, p1_contrib) if player == 0 else p0_contrib
            new_p1 = max(p0_contrib, p1_contrib) if player == 1 else p1_contrib
            self._enumerate_round2(actions + ['c'], 1 - player, new_p0, new_p1, num_bets)
            if num_bets < MAX_RAISES:
                if player == 0:
                    new_p0 = max(p0_contrib, p1_contrib) + ROUND2_BET
                else:
                    new_p1 = max(p0_contrib, p1_contrib) + ROUND2_BET
                self._enumerate_round2(actions + ['b'], 1 - player,
                                      new_p0 if player == 0 else p0_contrib,
                                      new_p1 if player == 1 else p1_contrib,
                                      num_bets + 1)
        else:
            self._enumerate_round2(actions + ['c'], 1 - player, p0_contrib, p1_contrib, num_bets)
            if num_bets < MAX_RAISES:
                if player == 0:
                    new_p0 = p0_contrib + ROUND2_BET
                else:
                    new_p1 = p1_contrib + ROUND2_BET
                self._enumerate_round2(actions + ['b'], 1 - player,
                                      new_p0 if player == 0 else p0_contrib,
                                      new_p1 if player == 1 else p1_contrib,
                                      num_bets + 1)

    def _to_numpy(self):
        """Convert to numpy arrays for fast vectorized access."""
        # Build infoset mapping: (round, rank, action_tuple) -> infoset_idx
        self.infoset_map = {}
        self.infoset_list = []
        infoset_idx = 0

        # Round 1 infosets (community unknown)
        for seq in self.round1_sequences:
            if not seq['is_terminal'] and not seq.get('to_round2', False):
                for rank in range(LEDUC_NUM_RANKS):
                    key = (1, rank, -1, seq['actions'])  # -1 for unknown community
                    if key not in self.infoset_map:
                        self.infoset_map[key] = infoset_idx
                        self.infoset_list.append({
                            'round': 1,
                            'rank': rank,
                            'community': -1,
                            'actions': seq['actions'],
                            'player': seq['player'],
                        })
                        infoset_idx += 1

        # Round 2 infosets (community known)
        for seq in self.round2_sequences:
            if not seq['is_terminal']:
                for rank in range(LEDUC_NUM_RANKS):
                    for comm in range(LEDUC_NUM_RANKS):
                        key = (2, rank, comm, seq['actions'])
                        if key not in self.infoset_map:
                            self.infoset_map[key] = infoset_idx
                            self.infoset_list.append({
                                'round': 2,
                                'rank': rank,
                                'community': comm,
                                'actions': seq['actions'],
                                'player': seq['player'],
                            })
                            infoset_idx += 1

        self.num_infosets = infoset_idx

        # Count actions per infoset
        self.infoset_num_actions = np.zeros(self.num_infosets, dtype=np.int32)
        self.infoset_player = np.zeros(self.num_infosets, dtype=np.int32)
        self.infoset_rank = np.zeros(self.num_infosets, dtype=np.int32)

        for idx, info in enumerate(self.infoset_list):
            self.infoset_player[idx] = info['player']
            self.infoset_rank[idx] = info['rank']

            # Determine number of actions
            actions = info['actions']
            facing_bet = self._facing_bet(list(actions))
            num_bets = sum(1 for a in actions if a == 'b')

            if facing_bet:
                self.infoset_num_actions[idx] = 3 if num_bets < MAX_RAISES else 2
            else:
                self.infoset_num_actions[idx] = 2 if num_bets < MAX_RAISES else 1


class SemiVectorLeducFast:
    """
    Fast Semi-Vector MCCFR for Leduc.

    Key optimizations:
    1. Pre-computed game tree
    2. Vectorized strategy computation
    3. Vectorized regret updates across all deals
    """

    def __init__(
        self,
        game: Game,
        sample_boards: bool = False,
        num_board_samples: int = 1
    ):
        if not isinstance(game, LeducPoker):
            raise ValueError("Only LeducPoker is supported")

        self.game = game
        self.sample_boards = sample_boards
        self.num_board_samples = num_board_samples

        # Build optimized game tree
        self._tree = LeducGameTree()
        self.num_infosets = self._tree.num_infosets
        self.max_actions = 3
        self.num_private_deals = LEDUC_NUM_PRIVATE_DEALS

        # Get all deals
        self._all_deals = np.array(get_all_private_deals(), dtype=np.int32)
        self._deal_ranks = np.array([
            [card_to_rank(h), card_to_rank(v)]
            for h, v in get_all_private_deals()
        ], dtype=np.int32)

        # Initialize regrets: (num_infosets, max_actions, num_deals)
        self._cumulative_regret = np.zeros(
            (self.num_infosets, self.max_actions, self.num_private_deals),
            dtype=np.float32
        )
        self._cumulative_strategy = np.zeros(
            (self.num_infosets, self.max_actions, self.num_private_deals),
            dtype=np.float32
        )

        self.iterations = 0

        # Pre-compute deal-to-infoset mappings
        self._precompute_mappings()

    def _precompute_mappings(self):
        """Pre-compute which deals map to which infosets."""
        # For each infoset, which deals are relevant (based on rank match)
        self._infoset_deal_mask = np.zeros(
            (self.num_infosets, self.num_private_deals), dtype=bool
        )

        for h_idx in range(self.num_infosets):
            player = self._tree.infoset_player[h_idx]
            rank = self._tree.infoset_rank[h_idx]

            for deal_idx in range(self.num_private_deals):
                deal_rank = self._deal_ranks[deal_idx, player]
                if deal_rank == rank:
                    self._infoset_deal_mask[h_idx, deal_idx] = True

    def _get_strategy_vectorized(self) -> np.ndarray:
        """
        Get strategy for all infosets and deals vectorized.

        Returns: (num_infosets, max_actions, num_deals)
        """
        # Regret matching: positive regrets normalized
        positive_regrets = np.maximum(self._cumulative_regret, 0)

        # Sum over actions
        regret_sum = positive_regrets.sum(axis=1, keepdims=True)  # (num_infosets, 1, num_deals)

        # Uniform fallback
        uniform = np.ones_like(self._cumulative_regret)
        for h_idx in range(self.num_infosets):
            num_actions = self._tree.infoset_num_actions[h_idx]
            uniform[h_idx, :num_actions, :] = 1.0 / num_actions
            uniform[h_idx, num_actions:, :] = 0.0

        # Safe division
        safe_sum = np.where(regret_sum > 0, regret_sum, 1.0)
        strategy = np.where(regret_sum > 0, positive_regrets / safe_sum, uniform)

        # Zero out invalid actions
        for h_idx in range(self.num_infosets):
            num_actions = self._tree.infoset_num_actions[h_idx]
            strategy[h_idx, num_actions:, :] = 0.0

        return strategy

    def iterate(self, num_iterations: int = 1) -> None:
        """Run MCCFR iterations."""
        for _ in range(num_iterations):
            self._single_iteration()
            self.iterations += 1

    def _single_iteration(self) -> None:
        """
        Run a single iteration with vectorized operations.

        Still loops over deals and board cards, but inner computations
        are optimized.
        """
        t = self.iterations + 1
        strategy = self._get_strategy_vectorized()

        # For each deal
        for deal_idx, (hero_card, villain_card) in enumerate(get_all_private_deals()):
            # Get remaining cards for community
            remaining = [c for c in range(LEDUC_NUM_CARDS)
                        if c != hero_card and c != villain_card]

            if self.sample_boards:
                boards = np.random.choice(remaining,
                                         size=min(self.num_board_samples, len(remaining)),
                                         replace=False)
                board_weight = 1.0
            else:
                boards = remaining
                board_weight = 1.0 / len(remaining)

            deal_weight = 1.0 / self.num_private_deals

            for community_card in boards:
                weight = deal_weight * board_weight

                # Update regrets for both players
                for player in range(2):
                    self._update_regrets_fast(
                        hero_card, villain_card, community_card,
                        player, strategy, deal_idx, weight
                    )

        # Update cumulative strategy
        self._cumulative_strategy += t * strategy

    def _update_regrets_fast(
        self,
        hero_card: int,
        villain_card: int,
        community_card: int,
        player: int,
        strategy: np.ndarray,
        deal_idx: int,
        weight: float
    ):
        """
        Update regrets using pre-computed tree structure.

        Uses iterative traversal instead of recursion.
        """
        hero_rank = card_to_rank(hero_card)
        villain_rank = card_to_rank(villain_card)
        community_rank = card_to_rank(community_card)

        player_rank = hero_rank if player == 0 else villain_rank
        opp_rank = villain_rank if player == 0 else hero_rank

        # Compute terminal values
        hero_hand = 100 + hero_rank if hero_rank == community_rank else hero_rank
        villain_hand = 100 + villain_rank if villain_rank == community_rank else villain_rank

        # Round 1 traversal
        r1_values = {}  # action_tuple -> (ev, reach_self, reach_opp)

        for seq in reversed(self._tree.round1_sequences):
            actions = seq['actions']

            if seq['is_terminal']:
                # Fold
                fold_player = seq['fold_player']
                if fold_player == 0:
                    ev = (seq['p1_contrib'], -seq['p1_contrib'])
                else:
                    ev = (-seq['p0_contrib'], seq['p0_contrib'])
                r1_values[actions] = (ev[player], 1.0, 1.0)

            elif seq.get('to_round2', False):
                # Continues to round 2 - compute expected value over round 2
                r2_ev = self._compute_round2_ev(
                    hero_rank, villain_rank, community_rank,
                    hero_hand, villain_hand,
                    seq['p0_contrib'], seq['p1_contrib'],
                    player, strategy, deal_idx
                )
                r1_values[actions] = (r2_ev, 1.0, 1.0)

            else:
                # Decision node
                acting_player = seq['player']
                acting_rank = hero_rank if acting_player == 0 else villain_rank

                # Get infoset
                key = (1, acting_rank, -1, actions)
                if key not in self._tree.infoset_map:
                    continue

                h_idx = self._tree.infoset_map[key]
                num_actions = self._tree.infoset_num_actions[h_idx]
                strat = strategy[h_idx, :num_actions, deal_idx]

                # Get child values
                facing_bet = self._tree._facing_bet(list(actions))
                num_bets = sum(1 for a in actions if a == 'b')

                if facing_bet:
                    child_actions = [('f',), ('c',)]
                    if num_bets < MAX_RAISES:
                        child_actions.append(('b',))
                else:
                    child_actions = [('c',)]
                    if num_bets < MAX_RAISES:
                        child_actions.append(('b',))

                action_values = []
                for a_tuple in child_actions:
                    child_key = actions + a_tuple
                    if child_key in r1_values:
                        action_values.append(r1_values[child_key][0])
                    else:
                        action_values.append(0.0)

                # Compute EV
                ev = sum(strat[i] * action_values[i] for i in range(len(action_values)))
                r1_values[actions] = (ev, 1.0, 1.0)

                # Update regrets if this is the player we're computing for
                if acting_player == player and acting_rank == player_rank:
                    for a_idx, av in enumerate(action_values):
                        instant_regret = av - ev
                        self._cumulative_regret[h_idx, a_idx, deal_idx] += weight * instant_regret

    def _compute_round2_ev(
        self,
        hero_rank: int,
        villain_rank: int,
        community_rank: int,
        hero_hand: int,
        villain_hand: int,
        p0_contrib: int,
        p1_contrib: int,
        player: int,
        strategy: np.ndarray,
        deal_idx: int
    ) -> float:
        """Compute expected value over round 2."""
        r2_values = {}

        for seq in reversed(self._tree.round2_sequences):
            actions = seq['actions']
            total_p0 = p0_contrib + seq['p0_contrib_delta']
            total_p1 = p1_contrib + seq['p1_contrib_delta']

            if seq['is_terminal']:
                if seq.get('is_showdown', False):
                    if hero_hand > villain_hand:
                        ev = (total_p1, -total_p1)
                    elif villain_hand > hero_hand:
                        ev = (-total_p0, total_p0)
                    else:
                        ev = (0.0, 0.0)
                else:
                    fold_player = seq['fold_player']
                    if fold_player == 0:
                        ev = (-total_p0, total_p0)
                    else:
                        ev = (total_p1, -total_p1)
                r2_values[actions] = ev[player]

            else:
                acting_player = seq['player']
                acting_rank = hero_rank if acting_player == 0 else villain_rank

                key = (2, acting_rank, community_rank, actions)
                if key not in self._tree.infoset_map:
                    r2_values[actions] = 0.0
                    continue

                h_idx = self._tree.infoset_map[key]
                num_actions = self._tree.infoset_num_actions[h_idx]
                strat = strategy[h_idx, :num_actions, deal_idx]

                facing_bet = self._tree._facing_bet(list(actions))
                num_bets = sum(1 for a in actions if a == 'b')

                if facing_bet:
                    child_actions = [('f',), ('c',)]
                    if num_bets < MAX_RAISES:
                        child_actions.append(('b',))
                else:
                    child_actions = [('c',)]
                    if num_bets < MAX_RAISES:
                        child_actions.append(('b',))

                action_values = []
                for a_tuple in child_actions:
                    child_key = actions + a_tuple
                    if child_key in r2_values:
                        action_values.append(r2_values[child_key])
                    else:
                        action_values.append(0.0)

                ev = sum(strat[i] * action_values[i] for i in range(len(action_values)))
                r2_values[actions] = ev

                # Update regrets
                player_rank = hero_rank if player == 0 else villain_rank
                if acting_player == player and acting_rank == player_rank:
                    for a_idx, av in enumerate(action_values):
                        instant_regret = av - ev
                        self._cumulative_regret[h_idx, a_idx, deal_idx] += (1.0 / len(get_all_private_deals()) / 4) * instant_regret

        return r2_values.get((), 0.0)

    def solve(self, iterations: int = 1000) -> None:
        """Solve the game."""
        self.iterate(iterations)

    @property
    def average_strategy(self) -> np.ndarray:
        """Get average strategy."""
        total = self._cumulative_strategy.sum(axis=1, keepdims=True)
        safe_total = np.where(total > 0, total, 1.0)
        avg = np.where(total > 0, self._cumulative_strategy / safe_total,
                      1.0 / self.max_actions)
        return avg

    def print_strategy(self):
        """Print strategy summary."""
        print(f"\nFast Leduc MCCFR after {self.iterations} iterations")
        print(f"Infosets: {self.num_infosets}")
