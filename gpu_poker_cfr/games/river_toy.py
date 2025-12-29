"""
River Toy Game - Simplified single-street Hold'em for testing.

Game structure:
- Fixed board (4 community cards: flop + turn)
- 2 hole cards per player
- 1 river card dealt
- Single betting round
- Showdown with 7-card hand evaluation

This is a stepping stone between Leduc and full NLHE.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from enum import Enum

# Standalone game - not using base Game class for simplicity
from .cards import (
    NUM_CARDS, card_rank, card_suit, card_name, cards_to_str,
    get_remaining_cards, make_card, RANK_A, RANK_K, RANK_Q, RANK_J
)
from .hand_eval import evaluate_hand, hand_value_to_name


# Game parameters
STARTING_POT = 10  # Chips in pot before river betting
BET_SIZE = 10       # Fixed bet size
MAX_RAISES = 2      # Maximum raises per round


class RiverAction(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3
    RAISE = 4


@dataclass
class RiverState:
    """State of a River toy game."""
    board: Tuple[int, ...]      # 4 cards (flop + turn)
    river: int                   # River card
    hole_cards: Tuple[Tuple[int, int], ...]  # (p0_cards, p1_cards)
    pot: int                     # Current pot size
    to_call: int                 # Amount to call
    num_bets: int                # Number of bets/raises made
    player: int                  # Current player (0 or 1)
    actions: Tuple[RiverAction, ...]  # Action history
    folded: int                  # -1 if no fold, else player who folded


class RiverToyGame:
    """River Toy Game for testing semi-vector MCCFR on larger game."""

    def __init__(self, board: Optional[List[int]] = None):
        """Initialize with optional fixed board.

        Args:
            board: 4 community cards (flop + turn). If None, uses default.
        """
        if board is None:
            # Default board: Ah Kd Qc 2s (rainbow flop with high cards)
            self.board = tuple([
                make_card(RANK_A, 2),  # Ah
                make_card(RANK_K, 1),  # Kd
                make_card(RANK_Q, 0),  # Qc
                make_card(2, 3),       # 4s
            ])
        else:
            self.board = tuple(board)

        assert len(self.board) == 4, "Board must have exactly 4 cards"

        # Get available cards (excluding board)
        self.available_cards = get_remaining_cards(list(self.board))
        self.num_available = len(self.available_cards)  # 48

        # Pre-compute all possible deals
        self._build_deals()

    def _build_deals(self):
        """Pre-compute all possible hole card + river combinations."""
        # For each river card, enumerate all hole card combinations
        self.deals = []  # (river, p0_cards, p1_cards)

        avail = self.available_cards

        for river_idx, river in enumerate(avail):
            remaining = [c for c in avail if c != river]

            # All ways to deal 2 cards to P0, then 2 cards to P1
            n = len(remaining)  # 47
            for i in range(n):
                for j in range(i + 1, n):
                    p0 = (remaining[i], remaining[j])
                    rest = [c for c in remaining if c not in p0]

                    for k in range(len(rest)):
                        for l in range(k + 1, len(rest)):
                            p1 = (rest[k], rest[l])
                            self.deals.append((river, p0, p1))

        self.num_deals = len(self.deals)

    def get_num_deals(self) -> int:
        """Return number of possible deals."""
        return self.num_deals

    def get_deal(self, deal_idx: int) -> Tuple[int, Tuple[int, int], Tuple[int, int]]:
        """Get deal by index: (river, p0_cards, p1_cards)."""
        return self.deals[deal_idx]

    def get_initial_state(self, deal_idx: int) -> RiverState:
        """Get initial state for a deal."""
        river, p0_cards, p1_cards = self.deals[deal_idx]
        return RiverState(
            board=self.board,
            river=river,
            hole_cards=(p0_cards, p1_cards),
            pot=STARTING_POT,
            to_call=0,
            num_bets=0,
            player=0,  # P0 acts first
            actions=(),
            folded=-1
        )

    def get_available_actions(self, state: RiverState) -> List[RiverAction]:
        """Get legal actions for current player."""
        if state.folded >= 0:
            return []

        if state.to_call > 0:
            # Facing a bet
            actions = [RiverAction.FOLD, RiverAction.CALL]
            if state.num_bets < MAX_RAISES:
                actions.append(RiverAction.RAISE)
        else:
            # Not facing a bet
            actions = [RiverAction.CHECK]
            if state.num_bets < MAX_RAISES:
                actions.append(RiverAction.BET)

        return actions

    def apply_action(self, state: RiverState, action: RiverAction) -> RiverState:
        """Apply action and return new state."""
        new_pot = state.pot
        new_to_call = 0
        new_num_bets = state.num_bets
        new_folded = state.folded

        if action == RiverAction.FOLD:
            new_folded = state.player
        elif action == RiverAction.CALL:
            new_pot += state.to_call
        elif action == RiverAction.CHECK:
            pass
        elif action == RiverAction.BET:
            new_pot += BET_SIZE
            new_to_call = BET_SIZE
            new_num_bets += 1
        elif action == RiverAction.RAISE:
            # Call + raise
            new_pot += state.to_call + BET_SIZE
            new_to_call = BET_SIZE
            new_num_bets += 1

        return RiverState(
            board=state.board,
            river=state.river,
            hole_cards=state.hole_cards,
            pot=new_pot,
            to_call=new_to_call,
            num_bets=new_num_bets,
            player=1 - state.player,
            actions=state.actions + (action,),
            folded=new_folded
        )

    def is_terminal(self, state: RiverState) -> bool:
        """Check if state is terminal."""
        if state.folded >= 0:
            return True

        actions = state.actions
        if len(actions) < 2:
            return False

        # Check-check or bet-call/raise-call
        if actions[-1] == RiverAction.CHECK and actions[-2] == RiverAction.CHECK:
            return True
        if actions[-1] == RiverAction.CALL:
            return True

        return False

    def get_payoffs(self, state: RiverState) -> Tuple[float, float]:
        """Get terminal payoffs for both players."""
        assert self.is_terminal(state)

        if state.folded >= 0:
            # Player folded
            if state.folded == 0:
                return -state.pot / 2, state.pot / 2
            else:
                return state.pot / 2, -state.pot / 2

        # Showdown
        board_river = list(state.board) + [state.river]
        hand0 = list(state.hole_cards[0]) + board_river
        hand1 = list(state.hole_cards[1]) + board_river

        value0 = evaluate_hand(hand0)
        value1 = evaluate_hand(hand1)

        half_pot = state.pot / 2
        if value0 > value1:
            return half_pot, -half_pot
        elif value1 > value0:
            return -half_pot, half_pot
        else:
            return 0.0, 0.0

    def get_infoset_key(self, state: RiverState, player: int) -> str:
        """Get information set key for a player.

        Player sees: own hole cards, board, river, action history
        """
        hole = state.hole_cards[player]
        hole_sorted = tuple(sorted(hole, reverse=True))
        actions_str = ''.join(a.name[0] for a in state.actions)
        return f"{card_name(hole_sorted[0])}{card_name(hole_sorted[1])}|{actions_str}"

    def get_infoset_key_int(self, state: RiverState, player: int) -> Tuple[int, int, int]:
        """Get integer-based infoset key: (hole_idx, action_seq_id)."""
        # Hole cards as sorted pair index
        hole = state.hole_cards[player]
        hole_sorted = tuple(sorted(hole, reverse=True))

        # Action sequence as integer
        action_id = 0
        for i, a in enumerate(state.actions):
            action_id = action_id * 5 + a.value

        return (hole_sorted[0], hole_sorted[1], action_id)


def enumerate_river_infosets(board: List[int]) -> Dict[str, List[Tuple[int, int]]]:
    """Enumerate all infosets for a given board.

    Returns dict mapping infoset_key -> list of (deal_idx, player)
    """
    game = RiverToyGame(board)
    infosets = {}

    def traverse(state: RiverState, deal_idx: int):
        if game.is_terminal(state):
            return

        player = state.player
        key = game.get_infoset_key(state, player)

        if key not in infosets:
            infosets[key] = []
        infosets[key].append((deal_idx, player))

        for action in game.get_available_actions(state):
            next_state = game.apply_action(state, action)
            traverse(next_state, deal_idx)

    # Traverse for subset of deals (full enumeration is huge)
    # For now, just get structure from first few deals
    for deal_idx in range(min(100, game.num_deals)):
        state = game.get_initial_state(deal_idx)
        traverse(state, deal_idx)

    return infosets


class RiverToyMini:
    """Mini River game with reduced deck for tractable solving.

    Uses only cards 9-A (24 cards) instead of full 52.
    This gives ~1M deals instead of 51M.
    """

    # Only use ranks 9, T, J, Q, K, A (indices 7-12)
    MIN_RANK = 7  # 9
    MAX_RANK = 12  # A

    def __init__(self, board: Optional[List[int]] = None):
        """Initialize with optional fixed board."""
        if board is None:
            # Default board: Ah Kd Qc 9s
            self.board = tuple([
                make_card(RANK_A, 2),   # Ah
                make_card(RANK_K, 1),   # Kd
                make_card(RANK_Q, 0),   # Qc
                make_card(7, 3),        # 9s
            ])
        else:
            self.board = tuple(board)

        # Only use cards with rank >= 9
        all_high_cards = [make_card(r, s)
                          for r in range(self.MIN_RANK, self.MAX_RANK + 1)
                          for s in range(4)]

        self.available_cards = [c for c in all_high_cards if c not in self.board]
        self.num_available = len(self.available_cards)

        self._build_deals()

    def _build_deals(self):
        """Build all possible deals."""
        avail = self.available_cards
        self.deals = []

        for river_idx, river in enumerate(avail):
            remaining = [c for c in avail if c != river]
            n = len(remaining)

            for i in range(n):
                for j in range(i + 1, n):
                    p0 = (remaining[i], remaining[j])
                    rest = [c for c in remaining if c not in p0]

                    for k in range(len(rest)):
                        for l in range(k + 1, len(rest)):
                            p1 = (rest[k], rest[l])
                            self.deals.append((river, p0, p1))

        self.num_deals = len(self.deals)

    def get_num_deals(self) -> int:
        return self.num_deals

    def get_deal(self, deal_idx: int) -> Tuple[int, Tuple[int, int], Tuple[int, int]]:
        return self.deals[deal_idx]

    def get_initial_state(self, deal_idx: int) -> RiverState:
        river, p0_cards, p1_cards = self.deals[deal_idx]
        return RiverState(
            board=self.board,
            river=river,
            hole_cards=(p0_cards, p1_cards),
            pot=STARTING_POT,
            to_call=0,
            num_bets=0,
            player=0,
            actions=(),
            folded=-1
        )

    def get_available_actions(self, state: RiverState) -> List[RiverAction]:
        if state.folded >= 0:
            return []

        if state.to_call > 0:
            actions = [RiverAction.FOLD, RiverAction.CALL]
            if state.num_bets < MAX_RAISES:
                actions.append(RiverAction.RAISE)
        else:
            actions = [RiverAction.CHECK]
            if state.num_bets < MAX_RAISES:
                actions.append(RiverAction.BET)

        return actions

    def apply_action(self, state: RiverState, action: RiverAction) -> RiverState:
        new_pot = state.pot
        new_to_call = 0
        new_num_bets = state.num_bets
        new_folded = state.folded

        if action == RiverAction.FOLD:
            new_folded = state.player
        elif action == RiverAction.CALL:
            new_pot += state.to_call
        elif action == RiverAction.CHECK:
            pass
        elif action == RiverAction.BET:
            new_pot += BET_SIZE
            new_to_call = BET_SIZE
            new_num_bets += 1
        elif action == RiverAction.RAISE:
            new_pot += state.to_call + BET_SIZE
            new_to_call = BET_SIZE
            new_num_bets += 1

        return RiverState(
            board=state.board,
            river=state.river,
            hole_cards=state.hole_cards,
            pot=new_pot,
            to_call=new_to_call,
            num_bets=new_num_bets,
            player=1 - state.player,
            actions=state.actions + (action,),
            folded=new_folded
        )

    def is_terminal(self, state: RiverState) -> bool:
        if state.folded >= 0:
            return True

        actions = state.actions
        if len(actions) < 2:
            return False

        if actions[-1] == RiverAction.CHECK and actions[-2] == RiverAction.CHECK:
            return True
        if actions[-1] == RiverAction.CALL:
            return True

        return False

    def get_payoffs(self, state: RiverState) -> Tuple[float, float]:
        assert self.is_terminal(state)

        if state.folded >= 0:
            if state.folded == 0:
                return -state.pot / 2, state.pot / 2
            else:
                return state.pot / 2, -state.pot / 2

        board_river = list(state.board) + [state.river]
        hand0 = list(state.hole_cards[0]) + board_river
        hand1 = list(state.hole_cards[1]) + board_river

        value0 = evaluate_hand(hand0)
        value1 = evaluate_hand(hand1)

        half_pot = state.pot / 2
        if value0 > value1:
            return half_pot, -half_pot
        elif value1 > value0:
            return -half_pot, half_pot
        else:
            return 0.0, 0.0

    def get_infoset_key(self, state: RiverState, player: int) -> str:
        hole = state.hole_cards[player]
        hole_sorted = tuple(sorted(hole, reverse=True))
        actions_str = ''.join(a.name[0] for a in state.actions)
        return f"{card_name(hole_sorted[0])}{card_name(hole_sorted[1])}|{actions_str}"


class RiverMicro:
    """Micro River game with just 8 cards (K, A with 4 suits).

    Perfect stepping stone between Leduc and larger games.
    - 8 cards total (K, A x 4 suits)
    - 4 cards on board
    - 4 remaining for river + hole cards
    - ~few hundred deals

    This is similar to Leduc in size but with real hand evaluation.
    """

    # Only use K and A
    RANKS = [RANK_K, RANK_A]

    def __init__(self, board: Optional[List[int]] = None):
        """Initialize with 4-card board from K/A cards."""
        if board is None:
            # Default: Ah Kh Kd Ad (two pairs on board)
            self.board = tuple([
                make_card(RANK_A, 2),  # Ah
                make_card(RANK_K, 2),  # Kh
                make_card(RANK_K, 1),  # Kd
                make_card(RANK_A, 1),  # Ad
            ])
        else:
            self.board = tuple(board)

        # All K/A cards
        all_cards = [make_card(r, s) for r in self.RANKS for s in range(4)]
        self.available_cards = [c for c in all_cards if c not in self.board]
        self.num_available = len(self.available_cards)

        self._build_deals()

    def _build_deals(self):
        """Build all deals: river + 2 hole cards each."""
        avail = self.available_cards
        self.deals = []

        # Each player gets 1 hole card (simplified)
        # 4 cards remaining: 1 river + 1 for P0 + 1 for P1 + 1 unused
        # Actually with 4 remaining cards and needing 5 (river + 2 + 2),
        # we can't do 2 hole cards each.

        # Simplify: each player gets 1 hole card
        for river_idx, river in enumerate(avail):
            remaining = [c for c in avail if c != river]

            for i, p0 in enumerate(remaining):
                for j, p1 in enumerate(remaining):
                    if i != j:
                        self.deals.append((river, (p0,), (p1,)))

        self.num_deals = len(self.deals)

    def get_num_deals(self) -> int:
        return self.num_deals

    def get_deal(self, deal_idx: int):
        return self.deals[deal_idx]

    def get_initial_state(self, deal_idx: int) -> RiverState:
        river, p0_cards, p1_cards = self.deals[deal_idx]
        return RiverState(
            board=self.board,
            river=river,
            hole_cards=(p0_cards, p1_cards),
            pot=STARTING_POT,
            to_call=0,
            num_bets=0,
            player=0,
            actions=(),
            folded=-1
        )

    def get_available_actions(self, state: RiverState) -> List[RiverAction]:
        if state.folded >= 0:
            return []

        if state.to_call > 0:
            actions = [RiverAction.FOLD, RiverAction.CALL]
            if state.num_bets < MAX_RAISES:
                actions.append(RiverAction.RAISE)
        else:
            actions = [RiverAction.CHECK]
            if state.num_bets < MAX_RAISES:
                actions.append(RiverAction.BET)

        return actions

    def apply_action(self, state: RiverState, action: RiverAction) -> RiverState:
        new_pot = state.pot
        new_to_call = 0
        new_num_bets = state.num_bets
        new_folded = state.folded

        if action == RiverAction.FOLD:
            new_folded = state.player
        elif action == RiverAction.CALL:
            new_pot += state.to_call
        elif action == RiverAction.CHECK:
            pass
        elif action == RiverAction.BET:
            new_pot += BET_SIZE
            new_to_call = BET_SIZE
            new_num_bets += 1
        elif action == RiverAction.RAISE:
            new_pot += state.to_call + BET_SIZE
            new_to_call = BET_SIZE
            new_num_bets += 1

        return RiverState(
            board=state.board,
            river=state.river,
            hole_cards=state.hole_cards,
            pot=new_pot,
            to_call=new_to_call,
            num_bets=new_num_bets,
            player=1 - state.player,
            actions=state.actions + (action,),
            folded=new_folded
        )

    def is_terminal(self, state: RiverState) -> bool:
        if state.folded >= 0:
            return True

        actions = state.actions
        if len(actions) < 2:
            return False

        if actions[-1] == RiverAction.CHECK and actions[-2] == RiverAction.CHECK:
            return True
        if actions[-1] == RiverAction.CALL:
            return True

        return False

    def get_payoffs(self, state: RiverState) -> Tuple[float, float]:
        assert self.is_terminal(state)

        if state.folded >= 0:
            if state.folded == 0:
                return -state.pot / 2, state.pot / 2
            else:
                return state.pot / 2, -state.pot / 2

        # Showdown - 6 cards each (4 board + river + 1 hole)
        board_river = list(state.board) + [state.river]
        # Pad to 7 cards using just the board+river (5 cards) + 2 duplicates for eval
        # Actually, let's evaluate with 6 cards (5 board + 1 hole) by treating as 7-card
        hand0 = list(state.hole_cards[0]) + board_river
        hand1 = list(state.hole_cards[1]) + board_river

        # Pad to 7 cards if needed
        while len(hand0) < 7:
            hand0.append(hand0[-1])  # Duplicate last card (won't affect eval)
        while len(hand1) < 7:
            hand1.append(hand1[-1])

        value0 = evaluate_hand(hand0)
        value1 = evaluate_hand(hand1)

        half_pot = state.pot / 2
        if value0 > value1:
            return half_pot, -half_pot
        elif value1 > value0:
            return -half_pot, half_pot
        else:
            return 0.0, 0.0

    def get_infoset_key(self, state: RiverState, player: int) -> str:
        hole = state.hole_cards[player]
        actions_str = ''.join(a.name[0] for a in state.actions)
        return f"{card_name(hole[0])}|{actions_str}"

    def get_hole_index(self, hole_card: int) -> int:
        """Get index of hole card in available cards."""
        if hole_card in self.available_cards:
            return self.available_cards.index(hole_card)
        # Map to 0-3 based on position in K/A deck
        all_cards = [make_card(r, s) for r in self.RANKS for s in range(4)]
        return all_cards.index(hole_card) if hole_card in all_cards else 0
