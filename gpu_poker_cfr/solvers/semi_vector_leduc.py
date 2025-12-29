"""
Semi-Vector MCCFR for Leduc Poker.

Leduc is larger than Kuhn and has a community card, making it ideal
for testing board sampling.

Architecture:
- Vectorize over ALL private card deals (30 for Leduc)
- Enumerate or sample community cards (4 possible per deal)
- For now: full enumeration to validate correctness
- Later: add sampling option

Leduc structure:
- 6-card deck: JJ, QQ, KK (2 of each rank)
- Each player gets 1 private card
- Round 1: betting (max 2 raises, bet size = 2)
- Community card dealt
- Round 2: betting (max 2 raises, bet size = 4)
- Showdown: pair (private == community) beats high card
"""

import numpy as np
from typing import Literal, Optional, Tuple, List, Dict
from itertools import permutations

from gpu_poker_cfr.games.base import Game, Player
from gpu_poker_cfr.games.leduc import (
    LeducPoker, JACK, QUEEN, KING, CARD_NAMES,
    FOLD, CALL, BET, ANTE, ROUND1_BET, ROUND2_BET, MAX_RAISES
)
from gpu_poker_cfr.engine.backend import get_backend


# =============================================================================
# Leduc Hand Indexing Constants
# =============================================================================
#
# DECK STRUCTURE:
# - 6 cards total: J0, J1, Q0, Q1, K0, K1 (two of each rank)
# - Card index = rank * 2 + suit (suit ∈ {0, 1})
# - Rank: J=0, Q=1, K=2
#
# PRIVATE DEALS:
# - 6 * 5 = 30 ordered pairs (hero_card, villain_card)
# - Cards are distinct (no duplicate card indices)
#
# COMMUNITY CARDS:
# - 4 remaining cards after dealing privates
# - Probability varies based on which cards are dealt
#
# =============================================================================

LEDUC_NUM_RANKS = 3       # J, Q, K
LEDUC_NUM_SUITS = 2       # Two of each rank
LEDUC_NUM_CARDS = 6       # Total cards in deck
LEDUC_NUM_PRIVATE_DEALS = 30  # 6 * 5 ordered pairs

# Card index to rank mapping
def card_to_rank(card_idx: int) -> int:
    """Convert card index (0-5) to rank (0-2)."""
    assert 0 <= card_idx < LEDUC_NUM_CARDS, f"Invalid card: {card_idx}"
    return card_idx // LEDUC_NUM_SUITS


def rank_to_name(rank: int) -> str:
    """Convert rank to name."""
    return CARD_NAMES[rank]


def card_to_name(card_idx: int) -> str:
    """Convert card index to name (e.g., 'J0', 'Q1')."""
    rank = card_to_rank(card_idx)
    suit = card_idx % LEDUC_NUM_SUITS
    return f"{CARD_NAMES[rank]}{suit}"


# All possible private deals
def get_all_private_deals() -> List[Tuple[int, int]]:
    """Get all 30 valid (hero_card, villain_card) private deals."""
    deals = []
    for hero in range(LEDUC_NUM_CARDS):
        for villain in range(LEDUC_NUM_CARDS):
            if hero != villain:
                deals.append((hero, villain))
    return deals


def get_remaining_cards(hero_card: int, villain_card: int) -> List[int]:
    """Get cards remaining in deck after private deal."""
    return [c for c in range(LEDUC_NUM_CARDS) if c != hero_card and c != villain_card]


def validate_private_deal(hero_card: int, villain_card: int) -> None:
    """Assert private deal is valid."""
    assert 0 <= hero_card < LEDUC_NUM_CARDS, f"hero_card={hero_card} invalid"
    assert 0 <= villain_card < LEDUC_NUM_CARDS, f"villain_card={villain_card} invalid"
    assert hero_card != villain_card, "Cards must be different"


# =============================================================================
# Hand Ranking
# =============================================================================

def hand_rank(private_card: int, community_card: int) -> int:
    """
    Compute hand rank. Higher is better.

    Pair (private rank == community rank) beats high card.
    Among pairs, higher rank wins.
    Among high cards, higher private card rank wins.
    """
    private_rank = card_to_rank(private_card)
    community_rank = card_to_rank(community_card)

    if private_rank == community_rank:
        # Pair: rank 100 + rank value
        return 100 + private_rank
    else:
        # High card: just private card rank
        return private_rank


# =============================================================================
# Semi-Vector MCCFR for Leduc
# =============================================================================

class SemiVectorLeducMCCFR:
    """
    Semi-Vector MCCFR solver for Leduc Poker.

    Key differences from Kuhn:
    1. Private deals: 30 instead of 6
    2. Community card: 4 remaining cards per deal
    3. Two betting rounds with different bet sizes
    4. Variable number of actions per infoset (2 or 3)
    """

    def __init__(
        self,
        game: Game,
        backend: Literal['numpy', 'cupy'] = 'numpy',
        sample_boards: bool = False,
        num_board_samples: int = 1
    ):
        """
        Initialize the semi-vector MCCFR solver for Leduc.

        Args:
            game: LeducPoker game instance
            backend: 'numpy' for CPU or 'cupy' for GPU
            sample_boards: If True, sample community cards instead of enumerating
            num_board_samples: Number of board samples per iteration (if sampling)
        """
        if not isinstance(game, LeducPoker):
            raise ValueError("Only LeducPoker is supported")

        self.game = game
        self.backend = get_backend(backend)
        self.sample_boards = sample_boards
        self.num_board_samples = num_board_samples

        # Build game tree for infoset structure
        self._tree = game.build_tree()

        # Infoset structure
        self.num_infosets = self._tree.num_infosets
        self.max_actions = 3  # Leduc can have fold/call/bet
        self.num_private_deals = LEDUC_NUM_PRIVATE_DEALS

        # Build infoset metadata
        self._build_infoset_metadata()

        # Initialize regrets and strategy sums
        # Shape: (num_infosets, max_actions, num_private_deals)
        # Note: We index by DEAL, not just hero's card, because
        # the opponent's card affects the community card distribution
        self._cumulative_regret = np.zeros(
            (self.num_infosets, self.max_actions, self.num_private_deals),
            dtype=np.float32
        )
        self._cumulative_strategy = np.zeros(
            (self.num_infosets, self.max_actions, self.num_private_deals),
            dtype=np.float32
        )

        # Cache valid actions per infoset
        self._infoset_num_actions = np.zeros(self.num_infosets, dtype=np.int32)
        for h_idx, infoset in enumerate(self._tree.infosets):
            self._infoset_num_actions[h_idx] = len(infoset.actions)

        # Iteration counter
        self.iterations = 0

    def _build_infoset_metadata(self):
        """Build metadata about each infoset."""
        self._infoset_player = np.zeros(self.num_infosets, dtype=np.int32)
        self._infoset_key = []
        self._infoset_private_rank = np.zeros(self.num_infosets, dtype=np.int32)
        self._infoset_community_rank = np.zeros(self.num_infosets, dtype=np.int32) - 1  # -1 = unknown
        self._infoset_round = np.zeros(self.num_infosets, dtype=np.int32)

        for h_idx, infoset in enumerate(self._tree.infosets):
            self._infoset_player[h_idx] = infoset.player.value
            self._infoset_key.append(infoset.key)

            # Parse key: "private:community:actions"
            parts = infoset.key.split(':')
            private_char = parts[0]
            community_char = parts[1] if len(parts) > 1 else '?'

            self._infoset_private_rank[h_idx] = {'J': JACK, 'Q': QUEEN, 'K': KING}[private_char]

            if community_char != '?':
                self._infoset_community_rank[h_idx] = {'J': JACK, 'Q': QUEEN, 'K': KING}[community_char]
                self._infoset_round[h_idx] = 2
            else:
                self._infoset_round[h_idx] = 1

    def _get_strategy(self, infoset_idx: int, deal_idx: int) -> np.ndarray:
        """
        Get current strategy for an infoset and deal via regret matching.

        Returns:
            strategy: Shape (num_actions,) - only valid actions
        """
        num_actions = self._infoset_num_actions[infoset_idx]
        regrets = self._cumulative_regret[infoset_idx, :num_actions, deal_idx]

        # Regret matching: positive regrets normalized
        positive_regrets = np.maximum(regrets, 0)
        regret_sum = positive_regrets.sum()

        if regret_sum > 0:
            return positive_regrets / regret_sum
        else:
            return np.ones(num_actions) / num_actions

    def _get_all_strategies(self) -> np.ndarray:
        """
        Get strategies for all infosets and deals.

        Returns:
            strategies: Shape (num_infosets, max_actions, num_private_deals)
        """
        strategies = np.zeros_like(self._cumulative_regret)

        for h_idx in range(self.num_infosets):
            num_actions = self._infoset_num_actions[h_idx]
            for deal_idx in range(self.num_private_deals):
                strategies[h_idx, :num_actions, deal_idx] = self._get_strategy(h_idx, deal_idx)

        return strategies

    def iterate(self, num_iterations: int = 1) -> None:
        """Run MCCFR iterations."""
        for _ in range(num_iterations):
            self._single_iteration()
            self.iterations += 1

    def _single_iteration(self) -> None:
        """
        Run a single semi-vector MCCFR iteration.

        For each private deal:
        1. Enumerate (or sample) community cards
        2. Compute counterfactual values
        3. Update regrets
        """
        t = self.iterations + 1

        # Get current strategies
        strategies = self._get_all_strategies()

        all_deals = get_all_private_deals()

        for deal_idx, (hero_card, villain_card) in enumerate(all_deals):
            deal_prob = 1.0 / len(all_deals)

            # Get remaining cards for community
            remaining = get_remaining_cards(hero_card, villain_card)

            if self.sample_boards:
                # Sample community cards
                sampled_boards = np.random.choice(
                    remaining,
                    size=min(self.num_board_samples, len(remaining)),
                    replace=False
                )
                board_prob = 1.0  # Importance sampling weight = 1 for uniform
            else:
                # Enumerate all community cards
                sampled_boards = remaining
                board_prob = 1.0 / len(remaining)

            for community_card in sampled_boards:
                # Update regrets for both players
                for player in range(2):
                    self._update_regrets_for_deal(
                        hero_card, villain_card, community_card,
                        player, strategies, deal_idx,
                        deal_prob * board_prob, t
                    )

        # Update cumulative strategy
        self._update_cumulative_strategy(strategies, t)

    def _update_regrets_for_deal(
        self,
        hero_card: int,
        villain_card: int,
        community_card: int,
        player: int,
        strategies: np.ndarray,
        deal_idx: int,
        weight: float,
        iteration: int
    ) -> None:
        """Update regrets for a specific deal and player."""
        # Determine which card this player holds
        player_card = hero_card if player == 0 else villain_card
        player_rank = card_to_rank(player_card)
        community_rank = card_to_rank(community_card)

        # Traverse game tree and compute CFVs
        cfvs = self._compute_cfv_recursive(
            hero_card, villain_card, community_card,
            player, strategies, deal_idx,
            round_num=1, actions=[],
            p0_reach=1.0, p1_reach=1.0,
            p0_contrib=ANTE, p1_contrib=ANTE
        )

        # Update regrets for this player's infosets
        for h_idx in range(self.num_infosets):
            if self._infoset_player[h_idx] != player:
                continue

            # Check if this infoset matches the player's card and community
            infoset_rank = self._infoset_private_rank[h_idx]
            if infoset_rank != player_rank:
                continue

            infoset_comm = self._infoset_community_rank[h_idx]
            if infoset_comm >= 0 and infoset_comm != community_rank:
                continue

            key = self._infoset_key[h_idx]
            if key not in cfvs:
                continue

            action_values, infoset_value = cfvs[key]
            num_actions = self._infoset_num_actions[h_idx]

            # Instant regret = action_value - infoset_value
            for a_idx in range(num_actions):
                instant_regret = action_values[a_idx] - infoset_value
                self._cumulative_regret[h_idx, a_idx, deal_idx] += weight * instant_regret

    def _compute_cfv_recursive(
        self,
        hero_card: int,
        villain_card: int,
        community_card: int,
        player: int,
        strategies: np.ndarray,
        deal_idx: int,
        round_num: int,
        actions: List,
        p0_reach: float,
        p1_reach: float,
        p0_contrib: int,
        p1_contrib: int
    ) -> Dict:
        """Recursively compute counterfactual values."""
        cfvs = {}

        # Check for terminal states
        terminal_value = self._get_terminal_value(
            hero_card, villain_card, community_card,
            actions, round_num, p0_contrib, p1_contrib
        )
        if terminal_value is not None:
            return cfvs

        # Check if we need to deal community card (transition from round 1 to 2)
        if round_num == 1 and self._is_round_over(actions):
            # Community card already fixed, continue to round 2
            return self._compute_cfv_recursive(
                hero_card, villain_card, community_card,
                player, strategies, deal_idx,
                round_num=2, actions=[],  # Reset actions for round 2
                p0_reach=p0_reach, p1_reach=p1_reach,
                p0_contrib=p0_contrib, p1_contrib=p1_contrib
            )

        # Determine who acts
        acting_player = self._get_acting_player(actions, round_num)
        if acting_player < 0:
            return cfvs

        acting_card = hero_card if acting_player == 0 else villain_card
        acting_rank = card_to_rank(acting_card)
        community_rank = card_to_rank(community_card)

        # Build infoset key
        infoset_key = self._make_infoset_key(acting_rank, community_rank if round_num == 2 else -1, actions, round_num)

        # Find infoset index
        h_idx = self._find_infoset(infoset_key)
        if h_idx is None:
            return cfvs

        # Get strategy for this infoset and deal
        num_actions = self._infoset_num_actions[h_idx]
        strategy = strategies[h_idx, :num_actions, deal_idx]

        # Get available actions
        available_actions = self._get_available_actions(actions, round_num)

        # Compute value for each action
        action_values = []
        for a_idx, action in enumerate(available_actions):
            new_actions = actions + [action]
            new_p0_contrib, new_p1_contrib = self._update_contributions(
                action, acting_player, p0_contrib, p1_contrib, round_num, actions
            )

            # Update reach probabilities
            if acting_player == 0:
                new_p0_reach = p0_reach * strategy[a_idx]
                new_p1_reach = p1_reach
            else:
                new_p0_reach = p0_reach
                new_p1_reach = p1_reach * strategy[a_idx]

            # Recurse
            child_cfvs = self._compute_cfv_recursive(
                hero_card, villain_card, community_card,
                player, strategies, deal_idx,
                round_num, new_actions,
                new_p0_reach, new_p1_reach,
                new_p0_contrib, new_p1_contrib
            )
            cfvs.update(child_cfvs)

            # Get value of this action
            action_value = self._get_action_value(
                hero_card, villain_card, community_card,
                player, strategies, deal_idx,
                round_num, new_actions,
                new_p0_reach, new_p1_reach,
                new_p0_contrib, new_p1_contrib
            )
            action_values.append(action_value)

        # Compute infoset value
        infoset_value = sum(strategy[a] * action_values[a] for a in range(len(action_values)))

        # Only store CFV for the player we're computing for
        if acting_player == player:
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
        community_card: int,
        player: int,
        strategies: np.ndarray,
        deal_idx: int,
        round_num: int,
        actions: List,
        p0_reach: float,
        p1_reach: float,
        p0_contrib: int,
        p1_contrib: int
    ) -> float:
        """Get expected value after taking actions."""
        terminal_value = self._get_terminal_value(
            hero_card, villain_card, community_card,
            actions, round_num, p0_contrib, p1_contrib
        )
        if terminal_value is not None:
            return terminal_value[player]

        # Check if transitioning to round 2
        if round_num == 1 and self._is_round_over(actions):
            return self._get_action_value(
                hero_card, villain_card, community_card,
                player, strategies, deal_idx,
                round_num=2, actions=[],
                p0_reach=p0_reach, p1_reach=p1_reach,
                p0_contrib=p0_contrib, p1_contrib=p1_contrib
            )

        acting_player = self._get_acting_player(actions, round_num)
        if acting_player < 0:
            return 0.0

        acting_card = hero_card if acting_player == 0 else villain_card
        acting_rank = card_to_rank(acting_card)
        community_rank = card_to_rank(community_card)

        infoset_key = self._make_infoset_key(acting_rank, community_rank if round_num == 2 else -1, actions, round_num)
        h_idx = self._find_infoset(infoset_key)

        if h_idx is None:
            return 0.0

        num_actions = self._infoset_num_actions[h_idx]
        strategy = strategies[h_idx, :num_actions, deal_idx]
        available_actions = self._get_available_actions(actions, round_num)

        ev = 0.0
        for a_idx, action in enumerate(available_actions):
            new_actions = actions + [action]
            new_p0_contrib, new_p1_contrib = self._update_contributions(
                action, acting_player, p0_contrib, p1_contrib, round_num, actions
            )

            if acting_player == 0:
                new_p0 = p0_reach * strategy[a_idx]
                new_p1 = p1_reach
            else:
                new_p0 = p0_reach
                new_p1 = p1_reach * strategy[a_idx]

            action_ev = self._get_action_value(
                hero_card, villain_card, community_card,
                player, strategies, deal_idx,
                round_num, new_actions,
                new_p0, new_p1,
                new_p0_contrib, new_p1_contrib
            )
            ev += strategy[a_idx] * action_ev

        return ev

    def _make_infoset_key(self, private_rank: int, community_rank: int, actions: List, round_num: int) -> str:
        """Build infoset key matching tree format."""
        private_char = CARD_NAMES[private_rank]
        community_char = CARD_NAMES[community_rank] if community_rank >= 0 else '?'
        action_str = ''.join(actions)
        return f"{private_char}:{community_char}:{action_str}"

    def _find_infoset(self, key: str) -> Optional[int]:
        """Find infoset index by key."""
        for i, k in enumerate(self._infoset_key):
            if k == key:
                return i
        return None

    def _get_terminal_value(
        self,
        hero_card: int,
        villain_card: int,
        community_card: int,
        actions: List,
        round_num: int,
        p0_contrib: int,
        p1_contrib: int
    ) -> Optional[Tuple[float, float]]:
        """Get terminal value if actions lead to terminal state."""
        if not actions:
            return None

        last_action = actions[-1]

        # Fold
        if last_action == 'f':
            acting = self._get_acting_player(actions[:-1], round_num)
            if acting == 0:
                # P1 folded, P2 wins
                return (float(-p0_contrib), float(p0_contrib))
            else:
                # P2 folded, P1 wins
                return (float(p1_contrib), float(-p1_contrib))

        # Check if round is over (both checked or call after bet)
        if self._is_round_over(actions):
            if round_num == 2:
                # Showdown
                hero_rank = hand_rank(hero_card, community_card)
                villain_rank = hand_rank(villain_card, community_card)

                if hero_rank > villain_rank:
                    return (float(p1_contrib), float(-p1_contrib))
                elif villain_rank > hero_rank:
                    return (float(-p0_contrib), float(p0_contrib))
                else:
                    return (0.0, 0.0)  # Tie
            # Round 1 over, not terminal

        return None

    def _is_round_over(self, actions: List) -> bool:
        """Check if betting round is complete."""
        if not actions:
            return False

        # Check-check
        if len(actions) >= 2 and actions[-1] == 'c' and actions[-2] == 'c':
            # Make sure neither was a call of a bet
            # cc at start means check-check
            if len(actions) == 2:
                return True

        # Bet followed by call
        for i, a in enumerate(actions):
            if a == 'b':
                # Look for matching call
                if i + 1 < len(actions) and actions[i + 1] == 'c':
                    # This might be it, but could be raise-call pattern
                    # Need to track more carefully
                    pass

        # Simplified: count bets and calls
        # Round ends when: cc (check-check) or last action is call after bet
        if len(actions) >= 2:
            if actions[-1] == 'c':
                # Check if previous non-c action was 'b'
                for i in range(len(actions) - 2, -1, -1):
                    if actions[i] == 'b':
                        return True
                    elif actions[i] == 'c':
                        continue
                    else:
                        break
                # Or check-check
                if all(a == 'c' for a in actions):
                    return True

        return False

    def _get_acting_player(self, actions: List, round_num: int) -> int:
        """Get which player acts given action history."""
        # In each round, P1 acts first
        # Actions alternate between players
        return len(actions) % 2

    def _get_available_actions(self, actions: List, round_num: int) -> List[str]:
        """Get available actions given history."""
        # Count raises in this round
        num_bets = sum(1 for a in actions if a == 'b')

        # Check if facing a bet
        facing_bet = False
        for a in reversed(actions):
            if a == 'b':
                facing_bet = True
                break
            elif a == 'c':
                continue
            else:
                break

        # Determine last aggressor
        if not actions or all(a == 'c' for a in actions):
            facing_bet = False
        elif actions[-1] == 'b':
            facing_bet = True
        elif len(actions) >= 2 and actions[-2] == 'b' and actions[-1] == 'c':
            facing_bet = False  # Bet was called

        # Recompute facing_bet more carefully
        facing_bet = False
        call_count_after_last_bet = 0
        for a in actions:
            if a == 'b':
                facing_bet = True
                call_count_after_last_bet = 0
            elif a == 'c' and facing_bet:
                call_count_after_last_bet += 1
                if call_count_after_last_bet >= 1:
                    facing_bet = False

        if facing_bet:
            if num_bets < MAX_RAISES:
                return ['f', 'c', 'b']  # fold, call, raise
            else:
                return ['f', 'c']  # fold, call (no more raises)
        else:
            if num_bets < MAX_RAISES:
                return ['c', 'b']  # check, bet
            else:
                return ['c']  # check only

    def _update_contributions(
        self,
        action: str,
        acting_player: int,
        p0_contrib: int,
        p1_contrib: int,
        round_num: int,
        prev_actions: List
    ) -> Tuple[int, int]:
        """Update pot contributions after an action."""
        bet_size = ROUND1_BET if round_num == 1 else ROUND2_BET

        if action == 'f':
            return p0_contrib, p1_contrib
        elif action == 'c':
            # Check or call
            if acting_player == 0:
                # P1 matches P2's contribution
                return max(p0_contrib, p1_contrib), p1_contrib
            else:
                return p0_contrib, max(p0_contrib, p1_contrib)
        elif action == 'b':
            # Bet or raise
            if acting_player == 0:
                new_contrib = max(p0_contrib, p1_contrib) + bet_size
                return new_contrib, p1_contrib
            else:
                new_contrib = max(p0_contrib, p1_contrib) + bet_size
                return p0_contrib, new_contrib

        return p0_contrib, p1_contrib

    def _update_cumulative_strategy(self, strategies: np.ndarray, iteration: int) -> None:
        """Update cumulative strategy sum."""
        self._cumulative_strategy += iteration * strategies

    def solve(self, iterations: int = 1000) -> None:
        """Solve the game by running MCCFR iterations."""
        self.iterate(iterations)

    @property
    def average_strategy(self) -> np.ndarray:
        """Get average strategy."""
        total = self._cumulative_strategy.sum(axis=1, keepdims=True)
        uniform = np.ones_like(self._cumulative_strategy) / self.max_actions

        # Safe division
        safe_total = np.where(total > 0, total, 1.0)
        avg = np.where(total > 0, self._cumulative_strategy / safe_total, uniform)
        return avg

    def print_strategy(self) -> None:
        """Print the average strategy for key infosets."""
        avg_strategy = self.average_strategy

        print(f"\nSemi-Vector Leduc MCCFR Strategy after {self.iterations} iterations:")
        print("-" * 70)

        # Group by round and private card
        for round_num in [1, 2]:
            print(f"\n{'='*20} Round {round_num} {'='*20}")
            for rank in [JACK, QUEEN, KING]:
                print(f"\n  Private: {CARD_NAMES[rank]}")
                for h_idx in range(self.num_infosets):
                    if self._infoset_private_rank[h_idx] != rank:
                        continue
                    if self._infoset_round[h_idx] != round_num:
                        continue

                    key = self._infoset_key[h_idx]
                    player = self._infoset_player[h_idx]
                    num_actions = self._infoset_num_actions[h_idx]

                    # Average over all deals where this player has this card
                    deal_mask = [
                        i for i, (h, v) in enumerate(get_all_private_deals())
                        if card_to_rank(h if player == 0 else v) == rank
                    ]

                    if deal_mask:
                        avg_strat = avg_strategy[h_idx, :num_actions, deal_mask].mean(axis=1)
                        actions = self._get_actions_for_infoset(h_idx)
                        action_strs = [f"{a}={avg_strat[i]:.2f}" for i, a in enumerate(actions[:num_actions])]
                        print(f"    P{player+1} [{key}]: {', '.join(action_strs)}")

    def _get_actions_for_infoset(self, h_idx: int) -> List[str]:
        """Get action names for an infoset."""
        num_actions = self._infoset_num_actions[h_idx]
        # Infer from tree
        infoset = self._tree.infosets[h_idx]
        return [a.name for a in infoset.actions]

    def exploitability(self) -> float:
        """Compute exploitability of the average strategy."""
        # Use vanilla CFR's exploitability as reference
        from gpu_poker_cfr.solvers.vanilla import VanillaCFR

        vanilla = VanillaCFR(self.game, backend='numpy')
        avg_strat = self.average_strategy
        vanilla_strat = np.zeros(vanilla.matrices.num_infoset_actions, dtype=np.float32)

        all_deals = get_all_private_deals()

        for h_idx in range(self.num_infosets):
            start = vanilla.matrices.infoset_action_offsets[h_idx]
            end = vanilla.matrices.infoset_action_offsets[h_idx + 1]
            num_actions = end - start

            # Average strategy over all relevant deals
            player = self._infoset_player[h_idx]
            rank = self._infoset_private_rank[h_idx]
            comm_rank = self._infoset_community_rank[h_idx]

            # For round 2 infosets, also filter by community card
            deal_indices = []
            for i, (h, v) in enumerate(all_deals):
                player_card = h if player == 0 else v
                if card_to_rank(player_card) == rank:
                    # For round 2, we need to consider community card
                    # But we average over all deals with matching private card
                    deal_indices.append(i)

            if deal_indices:
                # Get the strategy for valid action indices only
                # Shape: (num_actions, num_deals) then average over deals
                strat_slice = avg_strat[h_idx, :num_actions, :][:, deal_indices]
                avg = strat_slice.mean(axis=1)  # Average over deals
                vanilla_strat[start:end] = avg
            else:
                vanilla_strat[start:end] = 1.0 / num_actions

        vanilla._cumulative_strategy = vanilla.backend.dense_to_backend(vanilla_strat)
        vanilla.iterations = 1

        return vanilla.exploitability()
