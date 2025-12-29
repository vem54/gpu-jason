"""
52-card deck representation for Texas Hold'em.

Card encoding: card = rank * 4 + suit
- Ranks: 0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A
- Suits: 0=clubs, 1=diamonds, 2=hearts, 3=spades
- Card values: 0-51
"""

import numpy as np
from typing import List, Tuple

# Rank constants (0-indexed, 2 is lowest)
RANK_2 = 0
RANK_3 = 1
RANK_4 = 2
RANK_5 = 3
RANK_6 = 4
RANK_7 = 5
RANK_8 = 6
RANK_9 = 7
RANK_T = 8
RANK_J = 9
RANK_Q = 10
RANK_K = 11
RANK_A = 12

# Suit constants
CLUBS = 0
DIAMONDS = 1
HEARTS = 2
SPADES = 3

NUM_RANKS = 13
NUM_SUITS = 4
NUM_CARDS = 52

RANK_NAMES = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUIT_NAMES = ['c', 'd', 'h', 's']
SUIT_SYMBOLS = ['♣', '♦', '♥', '♠']


def make_card(rank: int, suit: int) -> int:
    """Create card from rank and suit."""
    return rank * 4 + suit


def card_rank(card: int) -> int:
    """Get rank of card (0-12)."""
    return card // 4


def card_suit(card: int) -> int:
    """Get suit of card (0-3)."""
    return card % 4


def card_name(card: int) -> str:
    """Get human-readable card name like 'As', 'Kh', '2c'."""
    return RANK_NAMES[card_rank(card)] + SUIT_NAMES[card_suit(card)]


def card_from_name(name: str) -> int:
    """Parse card name like 'As', 'Kh', '2c'."""
    rank = RANK_NAMES.index(name[0].upper())
    suit = SUIT_NAMES.index(name[1].lower())
    return make_card(rank, suit)


def cards_to_str(cards: List[int]) -> str:
    """Convert list of cards to string."""
    return ' '.join(card_name(c) for c in cards)


def get_all_cards() -> List[int]:
    """Get all 52 cards."""
    return list(range(52))


def get_remaining_cards(used: List[int]) -> List[int]:
    """Get cards not in used list."""
    used_set = set(used)
    return [c for c in range(52) if c not in used_set]


# Pre-compute card masks for fast set operations
def cards_to_mask(cards: List[int]) -> int:
    """Convert card list to 64-bit mask."""
    mask = 0
    for c in cards:
        mask |= (1 << c)
    return mask


def mask_to_cards(mask: int) -> List[int]:
    """Convert 64-bit mask to card list."""
    cards = []
    for c in range(52):
        if mask & (1 << c):
            cards.append(c)
    return cards


# Hand type constants (for hand evaluation)
HAND_HIGH_CARD = 0
HAND_PAIR = 1
HAND_TWO_PAIR = 2
HAND_THREE_OF_A_KIND = 3
HAND_STRAIGHT = 4
HAND_FLUSH = 5
HAND_FULL_HOUSE = 6
HAND_FOUR_OF_A_KIND = 7
HAND_STRAIGHT_FLUSH = 8

HAND_NAMES = [
    'High Card', 'Pair', 'Two Pair', 'Three of a Kind',
    'Straight', 'Flush', 'Full House', 'Four of a Kind', 'Straight Flush'
]
