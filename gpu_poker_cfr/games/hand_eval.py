"""
Fast 7-card poker hand evaluator.

Returns a hand value where higher = better.
Hand value encoding:
- Bits 20-23: Hand type (0-8, where 8=straight flush)
- Bits 0-19: Kickers/rank info for tie-breaking
"""

import numpy as np
from numba import njit
from typing import List, Tuple

from .cards import (
    card_rank, card_suit, NUM_RANKS, NUM_SUITS,
    HAND_HIGH_CARD, HAND_PAIR, HAND_TWO_PAIR, HAND_THREE_OF_A_KIND,
    HAND_STRAIGHT, HAND_FLUSH, HAND_FULL_HOUSE, HAND_FOUR_OF_A_KIND,
    HAND_STRAIGHT_FLUSH, HAND_NAMES, RANK_A
)


@njit(cache=True)
def _count_ranks(cards: np.ndarray) -> np.ndarray:
    """Count cards per rank."""
    counts = np.zeros(13, dtype=np.int32)
    for c in cards:
        counts[c // 4] += 1
    return counts


@njit(cache=True)
def _count_suits(cards: np.ndarray) -> np.ndarray:
    """Count cards per suit."""
    counts = np.zeros(4, dtype=np.int32)
    for c in cards:
        counts[c % 4] += 1
    return counts


@njit(cache=True)
def _get_straight_high(rank_mask: int) -> int:
    """Check for straight, return high card rank or -1.

    rank_mask: bit mask where bit i is set if rank i is present
    """
    # Check A-2-3-4-5 (wheel)
    wheel = 0b1000000001111  # A, 2, 3, 4, 5
    if (rank_mask & wheel) == wheel:
        return 3  # 5-high straight (rank of 5)

    # Check other straights (5 consecutive bits)
    for high in range(12, 3, -1):  # A down to 5
        mask = 0b11111 << (high - 4)
        if (rank_mask & mask) == mask:
            return high

    return -1


@njit(cache=True)
def _get_flush_suit(suit_counts: np.ndarray) -> int:
    """Return suit with 5+ cards, or -1."""
    for s in range(4):
        if suit_counts[s] >= 5:
            return s
    return -1


@njit(cache=True)
def _get_flush_ranks(cards: np.ndarray, flush_suit: int) -> np.ndarray:
    """Get ranks of cards in flush suit, sorted descending."""
    ranks = np.zeros(7, dtype=np.int32)
    count = 0
    for c in cards:
        if c % 4 == flush_suit:
            ranks[count] = c // 4
            count += 1

    # Sort descending (bubble sort for small array)
    for i in range(count):
        for j in range(i + 1, count):
            if ranks[j] > ranks[i]:
                ranks[i], ranks[j] = ranks[j], ranks[i]

    return ranks[:count]


@njit(cache=True)
def evaluate_7cards(cards: np.ndarray) -> int:
    """Evaluate 7-card hand, return hand value (higher = better)."""
    rank_counts = _count_ranks(cards)
    suit_counts = _count_suits(cards)

    # Build rank mask for straight detection
    rank_mask = 0
    for r in range(13):
        if rank_counts[r] > 0:
            rank_mask |= (1 << r)

    # Check for flush
    flush_suit = _get_flush_suit(suit_counts)

    if flush_suit >= 0:
        # Get flush card ranks
        flush_ranks = _get_flush_ranks(cards, flush_suit)

        # Build flush rank mask
        flush_rank_mask = 0
        for i in range(len(flush_ranks)):
            flush_rank_mask |= (1 << flush_ranks[i])

        # Check for straight flush
        straight_high = _get_straight_high(flush_rank_mask)
        if straight_high >= 0:
            return (HAND_STRAIGHT_FLUSH << 20) | straight_high

        # Regular flush - top 5 cards
        value = HAND_FLUSH << 20
        for i in range(5):
            value |= flush_ranks[i] << (16 - i * 4)
        return value

    # Count pairs, trips, quads
    quads_rank = -1
    trips_rank = -1
    pairs = np.zeros(2, dtype=np.int32)
    num_pairs = 0
    kickers = np.zeros(5, dtype=np.int32)
    num_kickers = 0

    # Scan ranks from high to low
    for r in range(12, -1, -1):
        count = rank_counts[r]
        if count == 4:
            quads_rank = r
        elif count == 3:
            if trips_rank < 0:
                trips_rank = r
            elif num_pairs < 2:
                pairs[num_pairs] = r
                num_pairs += 1
        elif count == 2:
            if num_pairs < 2:
                pairs[num_pairs] = r
                num_pairs += 1
        elif count == 1:
            if num_kickers < 5:
                kickers[num_kickers] = r
                num_kickers += 1

    # Four of a kind
    if quads_rank >= 0:
        # Best kicker from trips/pairs/kickers
        best_kicker = -1
        if trips_rank >= 0:
            best_kicker = trips_rank
        for i in range(num_pairs):
            if pairs[i] > best_kicker:
                best_kicker = pairs[i]
        if num_kickers > 0 and kickers[0] > best_kicker:
            best_kicker = kickers[0]
        return (HAND_FOUR_OF_A_KIND << 20) | (quads_rank << 4) | best_kicker

    # Full house
    if trips_rank >= 0 and num_pairs > 0:
        return (HAND_FULL_HOUSE << 20) | (trips_rank << 4) | pairs[0]

    # Check for straight
    straight_high = _get_straight_high(rank_mask)
    if straight_high >= 0:
        return (HAND_STRAIGHT << 20) | straight_high

    # Three of a kind
    if trips_rank >= 0:
        value = (HAND_THREE_OF_A_KIND << 20) | (trips_rank << 8)
        for i in range(min(2, num_kickers)):
            value |= kickers[i] << (4 - i * 4)
        return value

    # Two pair
    if num_pairs >= 2:
        value = (HAND_TWO_PAIR << 20) | (pairs[0] << 8) | (pairs[1] << 4)
        # Best kicker
        if num_kickers > 0:
            value |= kickers[0]
        return value

    # One pair
    if num_pairs == 1:
        value = (HAND_PAIR << 20) | (pairs[0] << 12)
        for i in range(min(3, num_kickers)):
            value |= kickers[i] << (8 - i * 4)
        return value

    # High card
    value = HAND_HIGH_CARD << 20
    for i in range(min(5, num_kickers)):
        value |= kickers[i] << (16 - i * 4)
    return value


@njit(cache=True)
def evaluate_6cards(cards: np.ndarray) -> int:
    """Evaluate 6-card hand by finding best 5-card combination.

    For turn all-in situations where we have 2 hole + 4 board.
    Evaluates all C(6,5)=6 possible 5-card hands and returns best.
    """
    best_value = -1

    # Try all 6 ways to pick 5 cards from 6
    # Exclude card i, evaluate remaining 5
    for exclude in range(6):
        # Build 5-card hand excluding one card
        # We can use a temporary 7-card array with duplicates
        # to reuse evaluate_7cards logic, but that's wasteful.
        # Instead, evaluate 5 cards directly.

        hand5 = np.zeros(5, dtype=np.int32)
        idx = 0
        for j in range(6):
            if j != exclude:
                hand5[idx] = cards[j]
                idx += 1

        value = _evaluate_5cards(hand5)
        if value > best_value:
            best_value = value

    return best_value


@njit(cache=True)
def _evaluate_5cards(cards: np.ndarray) -> int:
    """Evaluate exactly 5 cards. Used internally for 6-card eval."""
    # Count ranks and suits
    rank_counts = np.zeros(13, dtype=np.int32)
    suit_counts = np.zeros(4, dtype=np.int32)
    for c in cards:
        rank_counts[c // 4] += 1
        suit_counts[c % 4] += 1

    # Build rank mask
    rank_mask = 0
    for r in range(13):
        if rank_counts[r] > 0:
            rank_mask |= (1 << r)

    # Check for flush
    is_flush = False
    flush_suit = -1
    for s in range(4):
        if suit_counts[s] == 5:
            is_flush = True
            flush_suit = s
            break

    # Get sorted ranks for kickers
    sorted_ranks = np.zeros(5, dtype=np.int32)
    idx = 0
    for r in range(12, -1, -1):
        for _ in range(rank_counts[r]):
            if idx < 5:
                sorted_ranks[idx] = r
                idx += 1

    # Check for straight
    straight_high = _get_straight_high(rank_mask)

    # Straight flush
    if is_flush and straight_high >= 0:
        return (HAND_STRAIGHT_FLUSH << 20) | straight_high

    # Count pairs, trips, quads
    quads_rank = -1
    trips_rank = -1
    pair_rank = -1
    pair2_rank = -1
    kickers = np.zeros(5, dtype=np.int32)
    num_kickers = 0

    for r in range(12, -1, -1):
        count = rank_counts[r]
        if count == 4:
            quads_rank = r
        elif count == 3:
            trips_rank = r
        elif count == 2:
            if pair_rank < 0:
                pair_rank = r
            else:
                pair2_rank = r
        elif count == 1:
            if num_kickers < 5:
                kickers[num_kickers] = r
                num_kickers += 1

    # Four of a kind
    if quads_rank >= 0:
        return (HAND_FOUR_OF_A_KIND << 20) | (quads_rank << 4) | kickers[0]

    # Full house
    if trips_rank >= 0 and pair_rank >= 0:
        return (HAND_FULL_HOUSE << 20) | (trips_rank << 4) | pair_rank

    # Flush
    if is_flush:
        value = HAND_FLUSH << 20
        for i in range(5):
            value |= sorted_ranks[i] << (16 - i * 4)
        return value

    # Straight
    if straight_high >= 0:
        return (HAND_STRAIGHT << 20) | straight_high

    # Three of a kind
    if trips_rank >= 0:
        value = (HAND_THREE_OF_A_KIND << 20) | (trips_rank << 8)
        for i in range(min(2, num_kickers)):
            value |= kickers[i] << (4 - i * 4)
        return value

    # Two pair
    if pair_rank >= 0 and pair2_rank >= 0:
        value = (HAND_TWO_PAIR << 20) | (pair_rank << 8) | (pair2_rank << 4)
        if num_kickers > 0:
            value |= kickers[0]
        return value

    # One pair
    if pair_rank >= 0:
        value = (HAND_PAIR << 20) | (pair_rank << 12)
        for i in range(min(3, num_kickers)):
            value |= kickers[i] << (8 - i * 4)
        return value

    # High card
    value = HAND_HIGH_CARD << 20
    for i in range(5):
        value |= sorted_ranks[i] << (16 - i * 4)
    return value


def evaluate_hand(cards: List[int]) -> int:
    """Evaluate 7-card hand from Python list."""
    return evaluate_7cards(np.array(cards, dtype=np.int32))


def evaluate_6card_hand(cards: List[int]) -> int:
    """Evaluate 6-card hand from Python list (best 5 of 6)."""
    return evaluate_6cards(np.array(cards, dtype=np.int32))


def hand_value_to_type(value: int) -> int:
    """Extract hand type from hand value."""
    return value >> 20


def hand_value_to_name(value: int) -> str:
    """Get hand type name from hand value."""
    return HAND_NAMES[hand_value_to_type(value)]


def compare_hands(cards1: List[int], cards2: List[int]) -> int:
    """Compare two 7-card hands. Returns 1 if hand1 wins, -1 if hand2 wins, 0 if tie."""
    v1 = evaluate_hand(cards1)
    v2 = evaluate_hand(cards2)
    if v1 > v2:
        return 1
    elif v2 > v1:
        return -1
    return 0


# Vectorized evaluation for many hands
@njit(cache=True, parallel=True)
def evaluate_many_hands(hands: np.ndarray) -> np.ndarray:
    """Evaluate many 7-card hands.

    Args:
        hands: (N, 7) array of card indices

    Returns:
        (N,) array of hand values
    """
    n = hands.shape[0]
    values = np.zeros(n, dtype=np.int32)
    for i in range(n):
        values[i] = evaluate_7cards(hands[i])
    return values
