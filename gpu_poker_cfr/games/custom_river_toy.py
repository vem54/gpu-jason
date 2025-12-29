"""
Custom River Toy Game with specific ranges and board.
"""

import numpy as np
from typing import List, Tuple, Optional
from .cards import make_card, card_name, RANK_A, RANK_K, RANK_Q, RANK_J, RANK_T, RANK_5, CLUBS, SPADES, HEARTS


class CustomRiverToy:
    """
    Custom river toy game with:
    - Specific board
    - Specific ranges for each player
    - Custom pot/stack sizes
    - Check/All-in/Fold actions
    """

    def __init__(
        self,
        board: List[int],
        oop_range: List[Tuple[int, int]],  # List of (card1, card2) tuples
        ip_range: List[Tuple[int, int]],
        pot_size: int = 100,
        stack_size: int = 100,
    ):
        self.board = tuple(board)
        self.oop_range = oop_range
        self.ip_range = ip_range
        self.pot_size = pot_size
        self.stack_size = stack_size

        # Build all valid deals (OOP hand, IP hand combinations)
        self._build_deals()

        # Get all cards used
        all_cards = set(board)
        for h in oop_range:
            all_cards.add(h[0])
            all_cards.add(h[1])
        for h in ip_range:
            all_cards.add(h[0])
            all_cards.add(h[1])
        self.available_cards = sorted(all_cards)

    def _build_deals(self):
        """Build all valid deals where hands don't share cards."""
        self.deals = []
        board_set = set(self.board)

        for oop_hand in self.oop_range:
            oop_set = set(oop_hand)
            # Check OOP hand doesn't overlap with board
            if oop_set & board_set:
                continue

            for ip_hand in self.ip_range:
                ip_set = set(ip_hand)
                # Check IP hand doesn't overlap with board or OOP hand
                if ip_set & board_set or ip_set & oop_set:
                    continue

                self.deals.append((oop_hand, ip_hand))

        self.num_deals = len(self.deals)

    def get_deal(self, deal_idx: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return (oop_hand, ip_hand) for deal index."""
        return self.deals[deal_idx]

    def describe_deal(self, deal_idx: int) -> str:
        """Human-readable deal description."""
        oop, ip = self.deals[deal_idx]
        oop_str = f"{card_name(oop[0])}{card_name(oop[1])}"
        ip_str = f"{card_name(ip[0])}{card_name(ip[1])}"
        return f"OOP: {oop_str}, IP: {ip_str}"


def make_aa_kk_vs_ak_aq_game():
    """
    Create the specific game:
    - Board: As Ks Qs Th 5c
    - OOP: KK+, AT+ (AA, KK, AK, AQ, AJ, AT)
    - IP: KK+, JJ, 55, AQ+, JT (AA, KK, JJ, 55, AK, AQ, JT)
    - Pot: 100, Stack: 100
    """
    # Board: As Ks Qs Th 5c
    board = [
        make_card(RANK_A, SPADES),   # As
        make_card(RANK_K, SPADES),   # Ks
        make_card(RANK_Q, SPADES),   # Qs
        make_card(RANK_T, HEARTS),   # Th
        make_card(RANK_5, CLUBS),    # 5c
    ]
    board_set = set(board)

    def get_pair_combos(rank, blocked_suits):
        """Get all pair combos for a rank, excluding blocked suits."""
        available = [s for s in range(4) if s not in blocked_suits]
        combos = []
        for i, s1 in enumerate(available):
            for s2 in available[i+1:]:
                combos.append((make_card(rank, s1), make_card(rank, s2)))
        return combos

    def get_offsuit_combos(rank1, rank2, blocked_suits1, blocked_suits2):
        """Get all combos of two different ranks."""
        available1 = [s for s in range(4) if s not in blocked_suits1]
        available2 = [s for s in range(4) if s not in blocked_suits2]
        combos = []
        for s1 in available1:
            for s2 in available2:
                combos.append((make_card(rank1, s1), make_card(rank2, s2)))
        return combos

    # Blocked suits per rank on this board
    # As blocks A spade, Ks blocks K spade, Qs blocks Q spade, Th blocks T heart, 5c blocks 5 club
    blocked = {
        RANK_A: {SPADES},
        RANK_K: {SPADES},
        RANK_Q: {SPADES},
        RANK_J: set(),  # J not on board
        RANK_T: {HEARTS},
        RANK_5: {CLUBS},
    }

    # OOP range: KK+, AT+ (AA, KK, AT, AJ, AQ, AK)
    oop_range = []
    oop_range.extend(get_pair_combos(RANK_A, blocked[RANK_A]))  # AA
    oop_range.extend(get_pair_combos(RANK_K, blocked[RANK_K]))  # KK
    oop_range.extend(get_offsuit_combos(RANK_A, RANK_K, blocked[RANK_A], blocked[RANK_K]))  # AK
    oop_range.extend(get_offsuit_combos(RANK_A, RANK_Q, blocked[RANK_A], blocked[RANK_Q]))  # AQ
    oop_range.extend(get_offsuit_combos(RANK_A, RANK_J, blocked[RANK_A], blocked[RANK_J]))  # AJ
    oop_range.extend(get_offsuit_combos(RANK_A, RANK_T, blocked[RANK_A], blocked[RANK_T]))  # AT

    # IP range: KK+, JJ, 55, AQ+, JT (AA, KK, JJ, 55, AK, AQ, JT)
    ip_range = []
    ip_range.extend(get_pair_combos(RANK_A, blocked[RANK_A]))  # AA
    ip_range.extend(get_pair_combos(RANK_K, blocked[RANK_K]))  # KK
    ip_range.extend(get_pair_combos(RANK_J, blocked[RANK_J]))  # JJ
    ip_range.extend(get_pair_combos(RANK_5, blocked[RANK_5]))  # 55
    ip_range.extend(get_offsuit_combos(RANK_A, RANK_K, blocked[RANK_A], blocked[RANK_K]))  # AK
    ip_range.extend(get_offsuit_combos(RANK_A, RANK_Q, blocked[RANK_A], blocked[RANK_Q]))  # AQ
    ip_range.extend(get_offsuit_combos(RANK_J, RANK_T, blocked[RANK_J], blocked[RANK_T]))  # JT

    return CustomRiverToy(
        board=board,
        oop_range=oop_range,
        ip_range=ip_range,
        pot_size=100,
        stack_size=100
    )
