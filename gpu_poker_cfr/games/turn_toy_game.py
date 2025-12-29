"""
Turn Toy Game - Simplest turn game for testing.

Board: As Ks Qs Th (4 cards - turn)
OOP Range: AA, KK, AK, AQ, AJ, AT
IP Range: AA, KK, JJ, 55, AK, AQ, JT
Actions: Check / All-in only
Pot: 100, Stack: 100

Tree structure:
1. Turn betting: OOP Check/All-in -> IP responds
2. If not terminal after turn: CHANCE node (deal river)
3. River betting: Same structure
4. Showdown with 7-card evaluation
"""

import numpy as np
from typing import List, Tuple, Dict
from .cards import make_card, RANK_A, RANK_K, RANK_Q, RANK_J, RANK_T, RANK_5
from .cards import CLUBS, SPADES, HEARTS, DIAMONDS


# Node types
NODE_FOLD = 0
NODE_SHOWDOWN = 1
NODE_DECISION = 2
NODE_CHANCE = 3


class TurnGame:
    """
    Turn game with check/all-in only.

    Key additions vs river:
    - 4-card board (not 5)
    - Chance nodes for dealing river
    - Must track river outcomes for each deal
    """

    def __init__(
        self,
        board: List[int],  # 4 cards
        oop_range: List[Tuple[int, int]],
        ip_range: List[Tuple[int, int]],
        pot_size: float = 100,
        stack_size: float = 100,
    ):
        assert len(board) == 4, "Turn board must have exactly 4 cards"

        self.board = tuple(board)
        self.oop_range = oop_range
        self.ip_range = ip_range
        self.pot_size = pot_size
        self.stack_size = stack_size

        self._build_deals()
        self._compute_river_cards()

    def _build_deals(self):
        """Build all valid deals where hands don't share cards with board or each other."""
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
        print(f"Turn game: {self.num_deals} valid deals")

    def _compute_river_cards(self):
        """Compute all possible river cards (cards not on board)."""
        all_cards = set(range(52))
        board_set = set(self.board)
        self.possible_rivers = sorted(all_cards - board_set)  # 48 cards
        self.num_rivers = len(self.possible_rivers)
        print(f"Turn game: {self.num_rivers} possible river cards")

    def get_deal(self, deal_idx: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return (oop_hand, ip_hand) for deal index."""
        return self.deals[deal_idx]

    def get_valid_rivers(self, deal_idx: int) -> List[int]:
        """Get valid river cards for a specific deal (excluding cards in hands)."""
        oop_hand, ip_hand = self.deals[deal_idx]
        blocked = set(oop_hand) | set(ip_hand)
        return [r for r in self.possible_rivers if r not in blocked]


def build_turn_tree(game: TurnGame) -> Tuple[List[Dict], np.ndarray]:
    """
    Build the game tree for a turn game with chance nodes.

    Node types:
    - 0: FOLD (terminal)
    - 1: SHOWDOWN (terminal)
    - 2: DECISION
    - 3: CHANCE (river deal)

    Tree structure:
    - Root: OOP at turn (Check/All-in)
    - After OOP check: IP (Check/All-in)
    - After turn betting resolves (check-check or bet-call): CHANCE node
    - Each chance outcome leads to: River betting (same structure)
    - Terminal: Fold or Showdown
    """
    nodes = []
    max_actions = 2  # Check/All-in only

    pot = game.pot_size
    stack = game.stack_size

    # We'll build the tree with explicit tracking of which street we're on

    def add_decision_node(player, current_pot, oop_stack, ip_stack,
                          invested_oop, invested_ip, to_call, street):
        """Add a decision node. Returns node index."""
        player_stack = oop_stack if player == 0 else ip_stack

        if to_call == 0:
            # Can check or all-in
            if player_stack > 0:
                # Normal case: can check or go all-in
                actions = ['Check', 'All-in']
                amounts = [0, player_stack]
            else:
                # Stacks empty - can only check
                actions = ['Check']
                amounts = [0]
        else:
            # Facing bet - can fold or call
            actions = ['Fold', 'Call']
            amounts = [0, to_call]

        node = {
            'type': NODE_DECISION,
            'player': player,
            'pot': current_pot,
            'to_call': to_call,
            'stack_oop': oop_stack,
            'stack_ip': ip_stack,
            'invested_oop': invested_oop,
            'invested_ip': invested_ip,
            'actions': actions,
            'action_amounts': amounts,
            'street': street,  # 'turn' or 'river'
        }
        idx = len(nodes)
        nodes.append(node)
        return idx

    def add_terminal_node(node_type, current_pot, invested_oop, invested_ip, fold_player=None):
        """Add a terminal node (fold or showdown). Returns node index."""
        node = {
            'type': node_type,
            'pot': current_pot,
            'invested_oop': invested_oop,
            'invested_ip': invested_ip,
        }
        if node_type == NODE_FOLD:
            node['fold_player'] = fold_player
        idx = len(nodes)
        nodes.append(node)
        return idx

    def add_chance_node(current_pot, oop_stack, ip_stack, invested_oop, invested_ip):
        """Add a chance node for river deal. Returns node index."""
        node = {
            'type': NODE_CHANCE,
            'pot': current_pot,
            'stack_oop': oop_stack,
            'stack_ip': ip_stack,
            'invested_oop': invested_oop,
            'invested_ip': invested_ip,
            'num_outcomes': game.num_rivers,  # 48 possible rivers
        }
        idx = len(nodes)
        nodes.append(node)
        return idx

    # Build tree recursively
    # Note: We use BFS to build, then compute children

    # Start with root: OOP at turn
    root_idx = add_decision_node(
        player=0,
        current_pot=pot,
        oop_stack=stack,
        ip_stack=stack,
        invested_oop=pot/2,
        invested_ip=pot/2,
        to_call=0,
        street='turn'
    )

    # Process queue
    queue = [root_idx]
    child_indices = [[-1, -1] for _ in range(len(nodes))]  # Will grow as we add nodes

    processed = set()

    while queue:
        node_idx = queue.pop(0)
        if node_idx in processed:
            continue
        processed.add(node_idx)

        node = nodes[node_idx]

        if node['type'] in [NODE_FOLD, NODE_SHOWDOWN]:
            # Terminal - no children
            while len(child_indices) < len(nodes):
                child_indices.append([-1, -1])
            continue

        if node['type'] == NODE_CHANCE:
            # Chance node - leads to river betting
            # For simplicity in tree structure, chance node leads to single river root
            # (actual river card selection is handled in traversal)

            # River root: OOP to act with current pot/stacks
            river_root = add_decision_node(
                player=0,
                current_pot=node['pot'],
                oop_stack=node['stack_oop'],
                ip_stack=node['stack_ip'],
                invested_oop=node['invested_oop'],
                invested_ip=node['invested_ip'],
                to_call=0,
                street='river'
            )

            while len(child_indices) < len(nodes):
                child_indices.append([-1, -1])

            # Chance node has single "child" that represents all river outcomes
            child_indices[node_idx] = [river_root, -1]
            queue.append(river_root)
            continue

        # Decision node
        player = node['player']
        opponent = 1 - player
        street = node['street']

        children = [-1, -1]

        for action_idx, (action, amount) in enumerate(zip(node['actions'], node['action_amounts'])):
            new_pot = node['pot']
            new_stack_oop = node['stack_oop']
            new_stack_ip = node['stack_ip']
            new_invested_oop = node['invested_oop']
            new_invested_ip = node['invested_ip']

            if action == 'Fold':
                # Terminal - folder loses
                child_idx = add_terminal_node(
                    NODE_FOLD, new_pot, new_invested_oop, new_invested_ip, fold_player=player
                )

            elif action == 'Check':
                if player == 0:  # OOP checks
                    # IP to act
                    child_idx = add_decision_node(
                        player=1,
                        current_pot=new_pot,
                        oop_stack=new_stack_oop,
                        ip_stack=new_stack_ip,
                        invested_oop=new_invested_oop,
                        invested_ip=new_invested_ip,
                        to_call=0,
                        street=street
                    )
                    queue.append(child_idx)
                else:  # IP checks back
                    if street == 'turn':
                        # Go to chance node for river
                        child_idx = add_chance_node(
                            new_pot, new_stack_oop, new_stack_ip,
                            new_invested_oop, new_invested_ip
                        )
                        queue.append(child_idx)
                    else:
                        # River check-check = showdown
                        child_idx = add_terminal_node(
                            NODE_SHOWDOWN, new_pot, new_invested_oop, new_invested_ip
                        )

            elif action == 'Call':
                call_amount = amount
                if player == 0:
                    new_invested_oop += call_amount
                    new_stack_oop -= call_amount
                else:
                    new_invested_ip += call_amount
                    new_stack_ip -= call_amount
                new_pot += call_amount

                if street == 'turn':
                    # Call on turn -> chance node for river
                    child_idx = add_chance_node(
                        new_pot, new_stack_oop, new_stack_ip,
                        new_invested_oop, new_invested_ip
                    )
                    queue.append(child_idx)
                else:
                    # Call on river = showdown
                    child_idx = add_terminal_node(
                        NODE_SHOWDOWN, new_pot, new_invested_oop, new_invested_ip
                    )

            elif action == 'All-in':
                bet_amount = amount
                if player == 0:
                    new_invested_oop += bet_amount
                    new_stack_oop -= bet_amount
                    facing = bet_amount
                else:
                    new_invested_ip += bet_amount
                    new_stack_ip -= bet_amount
                    facing = bet_amount
                new_pot += bet_amount

                # Opponent faces the bet
                child_idx = add_decision_node(
                    player=opponent,
                    current_pot=new_pot,
                    oop_stack=new_stack_oop,
                    ip_stack=new_stack_ip,
                    invested_oop=new_invested_oop,
                    invested_ip=new_invested_ip,
                    to_call=facing,
                    street=street
                )
                queue.append(child_idx)

            children[action_idx] = child_idx

            while len(child_indices) < len(nodes):
                child_indices.append([-1, -1])

        child_indices[node_idx] = children

    # Ensure child_indices covers all nodes
    while len(child_indices) < len(nodes):
        child_indices.append([-1, -1])

    return nodes, np.array(child_indices, dtype=np.int32)


def make_turn_toy_game() -> TurnGame:
    """
    Create the turn toy game with standard setup.

    Board: As Ks Qs Th (4 cards - turn)
    OOP: AA, KK, AK, AQ, AJ, AT
    IP: AA, KK, JJ, 55, AK, AQ, JT
    Pot: 100, Stack: 100
    """
    board = [
        make_card(RANK_A, SPADES),
        make_card(RANK_K, SPADES),
        make_card(RANK_Q, SPADES),
        make_card(RANK_T, HEARTS),
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

    # Blocked suits (cards on board)
    blocked = {
        RANK_A: {SPADES},
        RANK_K: {SPADES},
        RANK_Q: {SPADES},
        RANK_J: set(),
        RANK_T: {HEARTS},
        RANK_5: set(),  # 5c not on board for turn
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

    print(f"OOP range: {len(oop_range)} combos")
    print(f"IP range: {len(ip_range)} combos")

    return TurnGame(
        board=board,
        oop_range=oop_range,
        ip_range=ip_range,
        pot_size=100,
        stack_size=100,
    )


def print_tree_structure(nodes, child_indices):
    """Print the tree structure for debugging."""
    type_names = {0: 'FOLD', 1: 'SHOWDOWN', 2: 'DECISION', 3: 'CHANCE'}
    print(f"\nTree has {len(nodes)} nodes:")

    for i, node in enumerate(nodes):
        t = type_names[node['type']]
        children = child_indices[i] if i < len(child_indices) else [-1, -1]

        if node['type'] == NODE_DECISION:
            print(f"  [{i}] {t} P{node['player']} {node['street']} "
                  f"pot={node['pot']:.0f} call={node['to_call']:.0f} "
                  f"actions={node['actions']} -> children {list(children)}")
        elif node['type'] == NODE_CHANCE:
            print(f"  [{i}] {t} pot={node['pot']:.0f} "
                  f"outcomes={node['num_outcomes']} -> children {list(children)}")
        elif node['type'] == NODE_FOLD:
            print(f"  [{i}] {t} folder=P{node['fold_player']} pot={node['pot']:.0f}")
        else:
            print(f"  [{i}] {t} pot={node['pot']:.0f}")


if __name__ == '__main__':
    game = make_turn_toy_game()
    nodes, child_indices = build_turn_tree(game)
    print_tree_structure(nodes, child_indices)
