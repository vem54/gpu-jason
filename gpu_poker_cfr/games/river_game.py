"""
Enhanced River Game with multiple bet sizes.

Supports configurable bet sizes as fractions of pot.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from .cards import make_card, card_name, card_from_name


class RiverGame:
    """
    River game with configurable bet sizes.

    Parameters:
    - board: 5 card board
    - oop_range: List of (card1, card2) tuples for OOP
    - ip_range: List of (card1, card2) tuples for IP
    - pot_size: Starting pot
    - stack_size: Effective stack (behind)
    - bet_sizes: List of bet sizes as fractions of pot (e.g., [0.5, 1.0] for 50% and pot)
    - raise_sizes: List of raise sizes as fractions of pot (e.g., [0.5, 1.0])
    - all_in: Whether to include all-in as an option
    """

    def __init__(
        self,
        board: List[int],
        oop_range: List[Tuple[int, int]],
        ip_range: List[Tuple[int, int]],
        pot_size: float = 100,
        stack_size: float = 100,
        bet_sizes: List[float] = [0.5, 1.0],  # 50% pot, 100% pot
        raise_sizes: List[float] = [1.0],      # Raise to 100% of pot
        all_in: bool = True,
    ):
        self.board = tuple(board)
        self.oop_range = oop_range
        self.ip_range = ip_range
        self.pot_size = pot_size
        self.stack_size = stack_size
        self.bet_sizes = bet_sizes
        self.raise_sizes = raise_sizes
        self.all_in = all_in

        self._build_deals()

    def _build_deals(self):
        """Build all valid deals where hands don't share cards."""
        self.deals = []
        board_set = set(self.board)

        for oop_hand in self.oop_range:
            oop_set = set(oop_hand)
            if oop_set & board_set:
                continue

            for ip_hand in self.ip_range:
                ip_set = set(ip_hand)
                if ip_set & board_set or ip_set & oop_set:
                    continue

                self.deals.append((oop_hand, ip_hand))

        self.num_deals = len(self.deals)

    def get_deal(self, deal_idx: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return (oop_hand, ip_hand) for deal index."""
        return self.deals[deal_idx]


def build_river_tree(game: RiverGame) -> Tuple[List[Dict], np.ndarray]:
    """
    Build the game tree for a river game.

    Returns:
    - nodes: List of node dictionaries
    - child_idx: numpy array of shape (num_nodes, max_actions) with child indices

    Node types:
    - 0: FOLD (terminal)
    - 1: SHOWDOWN (terminal)
    - 2: DECISION

    Each decision node has:
    - player: 0 (OOP) or 1 (IP)
    - pot: current pot size
    - to_call: amount to call (0 if no bet facing)
    - stack: remaining stack for acting player
    - actions: list of action names
    - action_amounts: list of amounts for each action
    """
    nodes = []
    max_actions = 6  # Check, Bet1, Bet2, All-in, Fold, Call, Raise...

    pot = game.pot_size
    stack = game.stack_size

    def get_bet_actions(current_pot: float, current_stack: float, facing_bet: float = 0):
        """Get available betting actions."""
        actions = []
        amounts = []

        if facing_bet == 0:
            # Can check
            actions.append('Check')
            amounts.append(0)

            # Bet sizes
            for size in game.bet_sizes:
                bet_amount = min(size * current_pot, current_stack)
                if bet_amount < current_stack:  # Don't duplicate all-in
                    actions.append(f'Bet {int(size*100)}%')
                    amounts.append(bet_amount)

            # All-in
            if game.all_in:
                actions.append('All-in')
                amounts.append(current_stack)
        else:
            # Facing a bet - can fold or call
            actions.append('Fold')
            amounts.append(0)

            call_amount = min(facing_bet, current_stack)
            actions.append('Call')
            amounts.append(call_amount)

            # Raise options (if we have more than just a call)
            if current_stack > facing_bet:
                for size in game.raise_sizes:
                    raise_to = min(facing_bet + size * (current_pot + facing_bet), current_stack)
                    if raise_to < current_stack:  # Don't duplicate all-in
                        actions.append(f'Raise {int(size*100)}%')
                        amounts.append(raise_to)

                if game.all_in:
                    actions.append('All-in')
                    amounts.append(current_stack)

        return actions, amounts

    # Use a queue-based approach to build tree
    node_queue = []  # (pot, oop_invested, ip_invested, to_call, player, parent_idx, action_idx)

    # Root node: OOP to act, no bet facing
    root_actions, root_amounts = get_bet_actions(pot, stack, 0)
    nodes.append({
        'type': 2,  # DECISION
        'player': 0,  # OOP
        'pot': pot,
        'to_call': 0,
        'stack_oop': stack,
        'stack_ip': stack,
        'invested_oop': pot / 2,  # Each player starts with half pot
        'invested_ip': pot / 2,
        'actions': root_actions,
        'action_amounts': root_amounts,
    })

    # Process queue iteratively
    node_idx = 0
    child_indices = []  # Will convert to numpy array later

    while node_idx < len(nodes):
        node = nodes[node_idx]

        if node['type'] != 2:  # Terminal node
            child_indices.append([-1] * max_actions)
            node_idx += 1
            continue

        children = [-1] * max_actions
        player = node['player']
        opponent = 1 - player

        for action_idx, (action, amount) in enumerate(zip(node['actions'], node['action_amounts'])):
            # Calculate new game state after action
            new_pot = node['pot']
            new_stack_oop = node['stack_oop']
            new_stack_ip = node['stack_ip']
            new_invested_oop = node['invested_oop']
            new_invested_ip = node['invested_ip']

            if action == 'Fold':
                # Terminal: folder loses their investment
                child_node = {
                    'type': 0,  # FOLD
                    'fold_player': player,
                    'pot': new_pot,
                    'invested_oop': new_invested_oop,
                    'invested_ip': new_invested_ip,
                }
            elif action == 'Check':
                if player == 0:  # OOP checks
                    # IP to act
                    ip_actions, ip_amounts = get_bet_actions(new_pot, new_stack_ip, 0)
                    child_node = {
                        'type': 2,
                        'player': 1,
                        'pot': new_pot,
                        'to_call': 0,
                        'stack_oop': new_stack_oop,
                        'stack_ip': new_stack_ip,
                        'invested_oop': new_invested_oop,
                        'invested_ip': new_invested_ip,
                        'actions': ip_actions,
                        'action_amounts': ip_amounts,
                    }
                else:  # IP checks back
                    # Showdown
                    child_node = {
                        'type': 1,  # SHOWDOWN
                        'pot': new_pot,
                        'invested_oop': new_invested_oop,
                        'invested_ip': new_invested_ip,
                    }
            elif action == 'Call':
                # Player calls, add to pot
                call_amount = amount
                if player == 0:
                    new_invested_oop += call_amount
                    new_stack_oop -= call_amount
                else:
                    new_invested_ip += call_amount
                    new_stack_ip -= call_amount
                new_pot += call_amount

                # Showdown after call
                child_node = {
                    'type': 1,  # SHOWDOWN
                    'pot': new_pot,
                    'invested_oop': new_invested_oop,
                    'invested_ip': new_invested_ip,
                }
            else:
                # Bet, Raise, or All-in
                bet_amount = amount
                if player == 0:
                    new_invested_oop += bet_amount
                    new_stack_oop -= bet_amount
                    facing = bet_amount - node['to_call']  # Additional amount opponent must call
                else:
                    new_invested_ip += bet_amount
                    new_stack_ip -= bet_amount
                    facing = bet_amount - node['to_call']

                new_pot += bet_amount

                # Opponent to act, facing bet
                opp_stack = new_stack_oop if opponent == 0 else new_stack_ip
                opp_actions, opp_amounts = get_bet_actions(new_pot, opp_stack, facing)

                child_node = {
                    'type': 2,
                    'player': opponent,
                    'pot': new_pot,
                    'to_call': facing,
                    'stack_oop': new_stack_oop,
                    'stack_ip': new_stack_ip,
                    'invested_oop': new_invested_oop,
                    'invested_ip': new_invested_ip,
                    'actions': opp_actions,
                    'action_amounts': opp_amounts,
                }

            # Add child node
            child_idx = len(nodes)
            children[action_idx] = child_idx
            nodes.append(child_node)

        child_indices.append(children)
        node_idx += 1

    return nodes, np.array(child_indices, dtype=np.int32)


def make_test_river_game(bet_sizes=[0.5], raise_sizes=[], all_in=True):
    """
    Create a test river game with the standard setup.

    Board: As Ks Qs Th 5c
    OOP: AA, KK, AK, AQ, AJ, AT
    IP: AA, KK, JJ, 55, AK, AQ, JT
    Pot: 100, Stack: 100
    """
    from .cards import make_card, RANK_A, RANK_K, RANK_Q, RANK_J, RANK_T, RANK_5, CLUBS, SPADES, HEARTS, DIAMONDS

    board = [
        make_card(RANK_A, SPADES),
        make_card(RANK_K, SPADES),
        make_card(RANK_Q, SPADES),
        make_card(RANK_T, HEARTS),
        make_card(RANK_5, CLUBS),
    ]
    board_set = set(board)

    def get_pair_combos(rank, blocked_suits):
        available = [s for s in range(4) if s not in blocked_suits]
        combos = []
        for i, s1 in enumerate(available):
            for s2 in available[i+1:]:
                combos.append((make_card(rank, s1), make_card(rank, s2)))
        return combos

    def get_offsuit_combos(rank1, rank2, blocked_suits1, blocked_suits2):
        available1 = [s for s in range(4) if s not in blocked_suits1]
        available2 = [s for s in range(4) if s not in blocked_suits2]
        combos = []
        for s1 in available1:
            for s2 in available2:
                combos.append((make_card(rank1, s1), make_card(rank2, s2)))
        return combos

    blocked = {
        RANK_A: {SPADES},
        RANK_K: {SPADES},
        RANK_Q: {SPADES},
        RANK_J: set(),
        RANK_T: {HEARTS},
        RANK_5: {CLUBS},
    }

    # OOP: AA, KK, AK, AQ, AJ, AT
    oop_range = []
    oop_range.extend(get_pair_combos(RANK_A, blocked[RANK_A]))
    oop_range.extend(get_pair_combos(RANK_K, blocked[RANK_K]))
    oop_range.extend(get_offsuit_combos(RANK_A, RANK_K, blocked[RANK_A], blocked[RANK_K]))
    oop_range.extend(get_offsuit_combos(RANK_A, RANK_Q, blocked[RANK_A], blocked[RANK_Q]))
    oop_range.extend(get_offsuit_combos(RANK_A, RANK_J, blocked[RANK_A], blocked[RANK_J]))
    oop_range.extend(get_offsuit_combos(RANK_A, RANK_T, blocked[RANK_A], blocked[RANK_T]))

    # IP: AA, KK, JJ, 55, AK, AQ, JT
    ip_range = []
    ip_range.extend(get_pair_combos(RANK_A, blocked[RANK_A]))
    ip_range.extend(get_pair_combos(RANK_K, blocked[RANK_K]))
    ip_range.extend(get_pair_combos(RANK_J, blocked[RANK_J]))
    ip_range.extend(get_pair_combos(RANK_5, blocked[RANK_5]))
    ip_range.extend(get_offsuit_combos(RANK_A, RANK_K, blocked[RANK_A], blocked[RANK_K]))
    ip_range.extend(get_offsuit_combos(RANK_A, RANK_Q, blocked[RANK_A], blocked[RANK_Q]))
    ip_range.extend(get_offsuit_combos(RANK_J, RANK_T, blocked[RANK_J], blocked[RANK_T]))

    return RiverGame(
        board=board,
        oop_range=oop_range,
        ip_range=ip_range,
        pot_size=100,
        stack_size=100,
        bet_sizes=bet_sizes,
        raise_sizes=raise_sizes,
        all_in=all_in,
    )
