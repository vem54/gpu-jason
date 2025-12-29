"""
Semi-Vector MCCFR (Monte Carlo CFR) Solver.

Key architecture:
- Vectorize over ALL hand combinations (no hand sampling)
- Sample only board cards (turn/river in real poker)
- For Kuhn: no board cards, so this is equivalent to full CFR

This provides:
- Low variance (no hand sampling)
- GPU parallelism over hands
- Memory efficiency (don't store full game tree)

For validation, we implement on Kuhn first where we can compare
against the known Nash equilibrium.
"""

import numpy as np
from typing import Literal, Optional, Tuple, List
from itertools import permutations

from gpu_poker_cfr.games.base import Game, Player
from gpu_poker_cfr.games.kuhn import KuhnPoker, JACK, QUEEN, KING, CARD_NAMES
from gpu_poker_cfr.engine.backend import get_backend, Backend


# =============================================================================
# Hand Indexing Constants (Step 2: Explicit documentation)
# =============================================================================
#
# ARCHITECTURE OVERVIEW:
# ----------------------
# Semi-vector MCCFR vectorizes over ALL hand combinations while sampling boards.
# For Kuhn (no board cards), this is equivalent to full CFR.
#
# MEMORY LAYOUT:
# - Regrets/Strategy: shape (num_infosets, max_actions, num_hands)
#   - Axis 0: Infoset index (12 for Kuhn)
#   - Axis 1: Action index within infoset
#   - Axis 2: Hand that the ACTING player holds
#
# KEY INVARIANTS:
# 1. Each infoset belongs to exactly one player
# 2. Regrets at infoset h are only valid for the player who acts at h
# 3. The hand dimension indexes the ACTING player's cards, not all deals
#
# =============================================================================

# For Kuhn Poker
KUHN_NUM_CARDS = 3  # J=0, Q=1, K=2
KUHN_NUM_HANDS = 3  # Each player can hold one of 3 cards
KUHN_NUM_DEALS = 6  # 3 * 2 = 6 valid (hero, villain) combinations

# Card constants (duplicated from games.kuhn for explicit documentation)
CARD_J = 0
CARD_Q = 1
CARD_K = 2

# Player indices
HERO = 0     # P1 (first to act)
VILLAIN = 1  # P2 (responds to hero)

# Reach probability columns (for compatibility with vectorized ops)
REACH_HERO = 0
REACH_VILLAIN = 1
REACH_CHANCE = 2
NUM_REACH_COLS = 3


def validate_card(card: int, name: str = "card") -> None:
    """Assert card is valid."""
    assert 0 <= card < KUHN_NUM_CARDS, \
        f"{name}={card} out of range [0, {KUHN_NUM_CARDS})"


def validate_deal(hero_card: int, villain_card: int) -> None:
    """Assert deal is valid (no duplicate cards)."""
    validate_card(hero_card, "hero_card")
    validate_card(villain_card, "villain_card")
    assert hero_card != villain_card, \
        f"Invalid deal: both players have card {hero_card}"


def deal_to_idx(hero_card: int, villain_card: int) -> int:
    """
    Convert (hero_card, villain_card) to deal index.

    Deal ordering: (J,Q)=0, (J,K)=1, (Q,J)=2, (Q,K)=3, (K,J)=4, (K,Q)=5
    """
    validate_deal(hero_card, villain_card)
    # Deals ordered as: (J,Q), (J,K), (Q,J), (Q,K), (K,J), (K,Q)
    deals = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
    return deals.index((hero_card, villain_card))


def idx_to_deal(idx: int) -> Tuple[int, int]:
    """
    Convert deal index to (hero_card, villain_card).

    Args:
        idx: Deal index in [0, KUHN_NUM_DEALS)

    Returns:
        (hero_card, villain_card) tuple
    """
    assert 0 <= idx < KUHN_NUM_DEALS, \
        f"deal_idx={idx} out of range [0, {KUHN_NUM_DEALS})"
    deals = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
    return deals[idx]


def get_all_deals() -> List[Tuple[int, int]]:
    """Get all valid (hero_card, villain_card) deals."""
    return [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]


# =============================================================================
# Semi-Vector MCCFR for Kuhn Poker
# =============================================================================

class SemiVectorMCCFR:
    """
    Semi-Vector MCCFR solver for Kuhn Poker.

    Architecture:
    - Regrets stored per (infoset, action, hero_hand): shape (num_infosets, max_actions, num_hands)
    - Strategy computed per hand
    - Vectorized updates across all hands simultaneously

    For Kuhn (no board cards), this is equivalent to full CFR and should
    converge to the same Nash equilibrium.
    """

    def __init__(
        self,
        game: Game,
        backend: Literal['numpy', 'cupy'] = 'numpy'
    ):
        """
        Initialize the semi-vector MCCFR solver.

        Args:
            game: Game to solve (currently only Kuhn supported)
            backend: 'numpy' for CPU or 'cupy' for GPU
        """
        if not isinstance(game, KuhnPoker):
            raise ValueError("Currently only KuhnPoker is supported")

        self.game = game
        self.backend = get_backend(backend)

        # Build game tree for infoset structure
        self._tree = game.build_tree()

        # Infoset structure
        self.num_infosets = self._tree.num_infosets
        self.max_actions = 2  # Kuhn always has 2 actions per infoset
        self.num_hands = KUHN_NUM_HANDS  # 3 hands per player
        self.num_deals = KUHN_NUM_DEALS  # 6 total deals

        # Build infoset metadata
        self._build_infoset_metadata()

        # Initialize regrets and strategy sums
        # Shape: (num_infosets, max_actions, num_hands)
        # Each infoset's regrets are indexed by the ACTING player's hand
        self._cumulative_regret = np.zeros(
            (self.num_infosets, self.max_actions, self.num_hands),
            dtype=np.float32
        )
        self._cumulative_strategy = np.zeros(
            (self.num_infosets, self.max_actions, self.num_hands),
            dtype=np.float32
        )

        # Iteration counter
        self.iterations = 0

    def _build_infoset_metadata(self):
        """Build metadata about each infoset."""
        self._infoset_player = np.zeros(self.num_infosets, dtype=np.int32)
        self._infoset_card = np.zeros(self.num_infosets, dtype=np.int32)
        self._infoset_key = []

        for h_idx, infoset in enumerate(self._tree.infosets):
            self._infoset_player[h_idx] = infoset.player.value
            self._infoset_key.append(infoset.key)

            # Extract card from infoset key (format: "X:..." where X is card)
            card_char = infoset.key[0]
            self._infoset_card[h_idx] = {'J': JACK, 'Q': QUEEN, 'K': KING}[card_char]

    def _get_strategy(self, infoset_idx: int) -> np.ndarray:
        """
        Get current strategy for an infoset via regret matching.

        Returns:
            strategy: Shape (num_actions, num_hands)
        """
        regrets = self._cumulative_regret[infoset_idx]  # (max_actions, num_hands)

        # Regret matching: positive regrets normalized
        positive_regrets = np.maximum(regrets, 0)
        regret_sum = positive_regrets.sum(axis=0, keepdims=True)  # (1, num_hands)

        # Avoid division by zero - use uniform where sum is 0
        uniform = np.ones_like(positive_regrets) / self.max_actions

        # Safe division: use uniform when regret_sum is 0
        safe_regret_sum = np.where(regret_sum > 0, regret_sum, 1.0)
        strategy = np.where(
            regret_sum > 0,
            positive_regrets / safe_regret_sum,
            uniform
        )

        return strategy

    def _get_all_strategies(self) -> np.ndarray:
        """
        Get strategies for all infosets.

        Returns:
            strategies: Shape (num_infosets, max_actions, num_hands)
        """
        strategies = np.zeros_like(self._cumulative_regret)
        for h_idx in range(self.num_infosets):
            strategies[h_idx] = self._get_strategy(h_idx)
        return strategies

    def iterate(self, num_iterations: int = 1) -> None:
        """Run MCCFR iterations."""
        for _ in range(num_iterations):
            self._single_iteration()
            self.iterations += 1

    def _single_iteration(self) -> None:
        """
        Run a single semi-vector MCCFR iteration.

        For each deal (hero_card, villain_card):
        1. Compute reach probabilities
        2. Compute counterfactual values
        3. Update regrets
        """
        t = self.iterations + 1

        # Get current strategies
        strategies = self._get_all_strategies()

        # For Kuhn with no board sampling, we iterate over all deals
        # In real poker, we would sample board cards here
        for deal_idx, (hero_card, villain_card) in enumerate(get_all_deals()):
            # Weight by deal probability (1/6 for Kuhn)
            deal_prob = 1.0 / self.num_deals

            # Traverse game tree for this deal
            for player in range(2):
                # Compute CFV and update regrets for this player
                self._update_regrets_for_deal(
                    hero_card, villain_card, player, strategies, deal_prob, t
                )

        # Update cumulative strategy (weighted by reach and iteration)
        self._update_cumulative_strategy(strategies, t)

    def _update_regrets_for_deal(
        self,
        hero_card: int,
        villain_card: int,
        player: int,
        strategies: np.ndarray,
        deal_prob: float,
        iteration: int
    ) -> None:
        """
        Update regrets for a specific deal and player.

        This traverses the game tree for the given deal and computes
        counterfactual values to update regrets.
        """
        # Determine which card this player holds
        player_card = hero_card if player == 0 else villain_card
        opponent_card = villain_card if player == 0 else hero_card

        # Traverse game tree and compute CFVs
        # For Kuhn, we can use a simple recursive traversal
        cfvs = self._compute_cfv_recursive(
            hero_card, villain_card, player, strategies,
            actions=[],
            p0_reach=1.0,
            p1_reach=1.0
        )

        # cfvs is a dict: infoset_key -> (action_values, infoset_value)
        # Update regrets for this player's infosets
        for h_idx in range(self.num_infosets):
            if self._infoset_player[h_idx] != player:
                continue

            if self._infoset_card[h_idx] != player_card:
                continue

            key = self._infoset_key[h_idx]
            if key not in cfvs:
                continue

            action_values, infoset_value = cfvs[key]

            # Instant regret = action_value - infoset_value
            for a_idx in range(len(action_values)):
                instant_regret = action_values[a_idx] - infoset_value
                # Update regret for this hand
                self._cumulative_regret[h_idx, a_idx, player_card] += deal_prob * instant_regret

    def _compute_cfv_recursive(
        self,
        hero_card: int,
        villain_card: int,
        player: int,  # Player we're computing CFV for
        strategies: np.ndarray,
        actions: List,
        p0_reach: float,
        p1_reach: float
    ) -> dict:
        """
        Recursively compute counterfactual values.

        Returns:
            dict mapping infoset_key -> (action_values, infoset_value)
        """
        cfvs = {}

        # Determine game state from action sequence
        # Use comma-separated format to match tree's infoset keys
        action_str = ','.join(a for a in actions)

        # Check for terminal states
        terminal_value = self._get_terminal_value(hero_card, villain_card, actions)
        if terminal_value is not None:
            # Terminal node - return value for the player
            return cfvs

        # Determine who acts
        acting_player = self._get_acting_player(actions)
        acting_card = hero_card if acting_player == 0 else villain_card

        # Build infoset key
        infoset_key = f"{CARD_NAMES[acting_card]}:{action_str}"

        # Find infoset index
        h_idx = None
        for i, key in enumerate(self._infoset_key):
            if key == infoset_key:
                h_idx = i
                break

        if h_idx is None:
            return cfvs

        # Get strategy for this infoset and hand
        strategy = strategies[h_idx, :, acting_card]  # (num_actions,)

        # Get available actions
        available_actions = self._get_available_actions(actions)

        # Compute value for each action
        action_values = []
        for a_idx, action in enumerate(available_actions):
            new_actions = actions + [action]

            # Update reach probabilities
            if acting_player == 0:
                new_p0_reach = p0_reach * strategy[a_idx]
                new_p1_reach = p1_reach
            else:
                new_p0_reach = p0_reach
                new_p1_reach = p1_reach * strategy[a_idx]

            # Recurse
            child_cfvs = self._compute_cfv_recursive(
                hero_card, villain_card, player, strategies,
                new_actions, new_p0_reach, new_p1_reach
            )
            cfvs.update(child_cfvs)

            # Get value of this action
            action_value = self._get_action_value(
                hero_card, villain_card, player,
                new_actions, new_p0_reach, new_p1_reach, strategies
            )
            action_values.append(action_value)

        # Compute infoset value (weighted by strategy)
        infoset_value = sum(strategy[a] * action_values[a] for a in range(len(action_values)))

        # Only store CFV for the player we're computing for
        if acting_player == player:
            # Weight by opponent's reach (counterfactual)
            opp_reach = p1_reach if player == 0 else p0_reach
            cfvs[infoset_key] = (
                [opp_reach * v for v in action_values],
                opp_reach * infoset_value
            )

        return cfvs

    def _get_action_value(
        self,
        hero_card: int,
        villain_card: int,
        player: int,
        actions: List,
        p0_reach: float,
        p1_reach: float,
        strategies: np.ndarray
    ) -> float:
        """Get expected value after taking actions."""
        terminal_value = self._get_terminal_value(hero_card, villain_card, actions)
        if terminal_value is not None:
            return terminal_value[player]

        # Not terminal, need to recurse
        acting_player = self._get_acting_player(actions)
        acting_card = hero_card if acting_player == 0 else villain_card

        action_str = ','.join(a for a in actions)
        infoset_key = f"{CARD_NAMES[acting_card]}:{action_str}"

        # Find infoset
        h_idx = None
        for i, key in enumerate(self._infoset_key):
            if key == infoset_key:
                h_idx = i
                break

        if h_idx is None:
            return 0.0

        strategy = strategies[h_idx, :, acting_card]
        available_actions = self._get_available_actions(actions)

        ev = 0.0
        for a_idx, action in enumerate(available_actions):
            new_actions = actions + [action]

            if acting_player == 0:
                new_p0 = p0_reach * strategy[a_idx]
                new_p1 = p1_reach
            else:
                new_p0 = p0_reach
                new_p1 = p1_reach * strategy[a_idx]

            action_ev = self._get_action_value(
                hero_card, villain_card, player,
                new_actions, new_p0, new_p1, strategies
            )
            ev += strategy[a_idx] * action_ev

        return ev

    def _get_terminal_value(
        self,
        hero_card: int,
        villain_card: int,
        actions: List
    ) -> Optional[Tuple[float, float]]:
        """
        Get terminal value if actions lead to terminal state.

        Returns:
            (hero_value, villain_value) or None if not terminal
        """
        if len(actions) == 0:
            return None

        action_str = ''.join(actions)

        # Kuhn terminal states:
        # cc -> showdown (both check)
        # bc -> P2 folds (P1 bet, P2 fold)  -- wait, 'c' after 'b' is call, not fold
        # Let me fix the action naming

        # Actions in our representation:
        # Round 1: P1 can check (c) or bet (b)
        # After check: P2 can check (c) or bet (b)
        # After bet: P2 can fold (f) or call (c)
        # After check-bet: P1 can fold (f) or call (c)

        # Terminal states:
        if action_str == 'cc':
            # Check-check: showdown, 1 chip each
            if hero_card > villain_card:
                return (1.0, -1.0)
            else:
                return (-1.0, 1.0)

        elif action_str == 'bf':
            # Bet-fold: P1 wins P2's ante
            return (1.0, -1.0)

        elif action_str == 'bc':
            # Bet-call: showdown, 2 chips each
            if hero_card > villain_card:
                return (2.0, -2.0)
            else:
                return (-2.0, 2.0)

        elif action_str == 'cbf':
            # Check-bet-fold: P2 wins P1's ante
            return (-1.0, 1.0)

        elif action_str == 'cbc':
            # Check-bet-call: showdown, 2 chips each
            if hero_card > villain_card:
                return (2.0, -2.0)
            else:
                return (-2.0, 2.0)

        return None

    def _get_acting_player(self, actions: List) -> int:
        """Get which player acts given action history."""
        if len(actions) == 0:
            return 0  # P1 acts first
        elif len(actions) == 1:
            return 1  # P2 acts second
        elif len(actions) == 2:
            if actions[0] == 'c' and actions[1] == 'b':
                return 0  # Check-bet, P1 responds
            else:
                return -1  # Terminal
        return -1

    def _get_available_actions(self, actions: List) -> List[str]:
        """Get available actions given history."""
        if len(actions) == 0:
            return ['c', 'b']  # Check or bet
        elif len(actions) == 1:
            if actions[0] == 'c':
                return ['c', 'b']  # Check or bet
            else:  # actions[0] == 'b'
                return ['f', 'c']  # Fold or call
        elif len(actions) == 2:
            if actions[0] == 'c' and actions[1] == 'b':
                return ['f', 'c']  # Fold or call
        return []

    def _update_cumulative_strategy(self, strategies: np.ndarray, iteration: int) -> None:
        """Update cumulative strategy sum (for average strategy computation)."""
        # For each infoset, weight strategy by reach probability
        # In semi-vector, reach is 1.0 for the player's own hands
        # We weight by iteration number (CFR+ style)
        self._cumulative_strategy += iteration * strategies

    def solve(self, iterations: int = 1000) -> None:
        """Solve the game by running MCCFR iterations."""
        self.iterate(iterations)

    @property
    def average_strategy(self) -> np.ndarray:
        """
        Get average strategy.

        Returns:
            Shape (num_infosets, max_actions, num_hands)
        """
        total = self._cumulative_strategy.sum(axis=1, keepdims=True)
        uniform = np.ones_like(self._cumulative_strategy) / self.max_actions

        avg = np.where(total > 0, self._cumulative_strategy / total, uniform)
        return avg

    def get_strategy_for_infoset(self, infoset_idx: int, hand: int) -> np.ndarray:
        """Get average strategy for a specific infoset and hand."""
        return self.average_strategy[infoset_idx, :, hand]

    def get_infoset_name(self, infoset_idx: int) -> str:
        """Get human-readable name for an infoset."""
        return self._infoset_key[infoset_idx]

    def print_strategy(self) -> None:
        """Print the average strategy for all infosets."""
        avg_strategy = self.average_strategy

        print(f"\nSemi-Vector MCCFR Strategy after {self.iterations} iterations:")
        print("-" * 60)

        for h_idx in range(self.num_infosets):
            player = self._infoset_player[h_idx]
            card = self._infoset_card[h_idx]
            key = self._infoset_key[h_idx]

            # Strategy for this infoset (should be same for all hands due to infoset structure)
            strat = avg_strategy[h_idx, :, card]

            actions = self._get_actions_for_infoset(key)
            action_strs = [f"{a}={strat[i]:.3f}" for i, a in enumerate(actions)]

            print(f"P{player+1} [{key}]: {', '.join(action_strs)}")

    def _get_actions_for_infoset(self, key: str) -> List[str]:
        """Get action names for an infoset."""
        action_part = key.split(':')[1] if ':' in key else ''

        if action_part == '':
            return ['c', 'b']  # P1 first action: check or bet
        elif action_part == 'c':
            return ['c', 'b']  # P2 after check: check or bet
        elif action_part == 'b':
            return ['f', 'c']  # P2 after bet: fold or call
        elif action_part == 'c,b':
            return ['f', 'c']  # P1 after check-bet: fold or call
        return ['?', '?']

    def exploitability(self) -> float:
        """
        Compute exploitability of the average strategy.

        For validation, this should converge to near 0 for Nash equilibrium.
        """
        # Simplified exploitability for Kuhn
        # Use the same approach as VanillaCFR
        from gpu_poker_cfr.solvers.vanilla import VanillaCFR

        # Create a vanilla solver and copy our strategy
        vanilla = VanillaCFR(self.game, backend='numpy')

        # Convert our strategy to vanilla format
        avg_strat = self.average_strategy
        vanilla_strat = np.zeros(vanilla.matrices.num_infoset_actions, dtype=np.float32)

        for h_idx in range(self.num_infosets):
            card = self._infoset_card[h_idx]
            start = vanilla.matrices.infoset_action_offsets[h_idx]
            end = vanilla.matrices.infoset_action_offsets[h_idx + 1]
            vanilla_strat[start:end] = avg_strat[h_idx, :end-start, card]

        # Use vanilla's exploitability calculation
        vanilla._cumulative_strategy = vanilla.backend.dense_to_backend(vanilla_strat)
        vanilla.iterations = 1

        return vanilla.exploitability()
