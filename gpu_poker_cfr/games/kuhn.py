"""
Kuhn Poker implementation.

Kuhn Poker is a simplified poker game:
- 3-card deck: Jack (J=0), Queen (Q=1), King (K=2)
- Each player antes 1 chip
- Each player is dealt one card
- Player 1 acts first: Check or Bet
- Betting round follows standard poker rules
- Higher card wins at showdown

Game tree has 58 nodes, 30 terminals, 12 infosets.
"""

from typing import Dict, List, Tuple, Optional
from itertools import permutations
import numpy as np

from .base import Game, GameTree, Node, Infoset, Action, Player


# Card values
JACK = 0
QUEEN = 1
KING = 2
CARD_NAMES = {JACK: 'J', QUEEN: 'Q', KING: 'K'}

# Actions
CHECK = Action(id=0, name='c')  # Check / Pass
BET = Action(id=1, name='b')    # Bet 1 chip
FOLD = Action(id=0, name='f')   # Fold (same id as check - context dependent)
CALL = Action(id=1, name='c')   # Call (same id as bet - context dependent)


class KuhnPoker(Game):
    """Kuhn Poker game implementation."""

    @property
    def name(self) -> str:
        return "kuhn_poker"

    @property
    def num_players(self) -> int:
        return 2

    def infoset_key(self, player: Player, history: Tuple) -> str:
        """
        Information set key for a player given history.

        Player knows: their own card + action history
        Player doesn't know: opponent's card
        """
        cards, actions = history
        my_card = cards[player.value]
        return f"{CARD_NAMES[my_card]}:{','.join(a.name for a in actions)}"

    def build_tree(self) -> GameTree:
        """Build the complete Kuhn Poker game tree."""
        nodes: List[Node] = []
        infosets: Dict[str, Infoset] = {}
        infoset_list: List[Infoset] = []

        node_id = 0
        infoset_id = 0

        def get_or_create_infoset(player: Player, history: Tuple, actions: Tuple[Action, ...]) -> int:
            nonlocal infoset_id
            key = self.infoset_key(player, history)
            if key not in infosets:
                infosets[key] = Infoset(
                    id=infoset_id,
                    player=player,
                    actions=actions,
                    node_ids=[],
                    key=key
                )
                infoset_list.append(infosets[key])
                infoset_id += 1
            return infosets[key].id

        def add_node(
            parent_id: Optional[int],
            player: Player,
            history: Tuple,
            action_from_parent: Optional[Action],
            depth: int,
            is_terminal: bool,
            utility: Optional[Tuple[float, float]],
            chance_prob: float,
            available_actions: Tuple[Action, ...]
        ) -> int:
            nonlocal node_id

            current_id = node_id
            node_id += 1

            # Get or create infoset for decision nodes
            infoset_id_for_node = None
            if not is_terminal and player in (Player.PLAYER_1, Player.PLAYER_2):
                infoset_id_for_node = get_or_create_infoset(player, history, available_actions)
                infosets[self.infoset_key(player, history)].node_ids.append(current_id)

            nodes.append(Node(
                id=current_id,
                parent_id=parent_id,
                player=player,
                infoset_id=infoset_id_for_node,
                actions=available_actions,
                action_from_parent=action_from_parent,
                depth=depth,
                is_terminal=is_terminal,
                utility=utility,
                chance_prob=chance_prob
            ))

            return current_id

        def terminal_utility(
            cards: Tuple[int, int],
            p1_contrib: int,
            p2_contrib: int,
            winner: Optional[int]
        ) -> Tuple[float, float]:
            """
            Calculate terminal utilities.

            Args:
                cards: (p1_card, p2_card)
                p1_contrib: chips P1 put in pot
                p2_contrib: chips P2 put in pot
                winner: 0 if P1 wins (fold), 1 if P2 wins (fold), None for showdown

            Returns:
                (p1_utility, p2_utility) - net gain/loss for each player
            """
            if winner is not None:
                # Someone folded - winner takes pot
                if winner == 0:
                    # P1 wins: gains P2's contribution
                    return (float(p2_contrib), float(-p2_contrib))
                else:
                    # P2 wins: gains P1's contribution
                    return (float(-p1_contrib), float(p1_contrib))
            else:
                # Showdown - higher card wins the pot
                if cards[0] > cards[1]:
                    # P1 wins: gains P2's contribution
                    return (float(p2_contrib), float(-p2_contrib))
                else:
                    # P2 wins: gains P1's contribution
                    return (float(-p1_contrib), float(p1_contrib))

        def build_subtree(
            parent_id: int,
            cards: Tuple[int, int],
            actions: Tuple[Action, ...],
            depth: int,
            chance_prob: float
        ):
            """Recursively build game tree from a position."""
            history = (cards, actions)

            # Determine game state from action sequence
            if len(actions) == 0:
                # Player 1's first action
                p1_actions = (Action(0, 'c'), Action(1, 'b'))  # check or bet
                nid = add_node(parent_id, Player.PLAYER_1, history, None, depth, False, None, 1.0, p1_actions)
                # Check branch
                build_subtree(nid, cards, actions + (Action(0, 'c'),), depth + 1, chance_prob)
                # Bet branch
                build_subtree(nid, cards, actions + (Action(1, 'b'),), depth + 1, chance_prob)

            elif len(actions) == 1:
                if actions[0].id == 0:  # P1 checked
                    # Player 2: check or bet
                    p2_actions = (Action(0, 'c'), Action(1, 'b'))
                    nid = add_node(parent_id, Player.PLAYER_2, history, actions[-1], depth, False, None, 1.0, p2_actions)
                    # Check branch -> showdown
                    build_subtree(nid, cards, actions + (Action(0, 'c'),), depth + 1, chance_prob)
                    # Bet branch
                    build_subtree(nid, cards, actions + (Action(1, 'b'),), depth + 1, chance_prob)
                else:  # P1 bet
                    # Player 2: fold or call
                    p2_actions = (Action(0, 'f'), Action(1, 'c'))
                    nid = add_node(parent_id, Player.PLAYER_2, history, actions[-1], depth, False, None, 1.0, p2_actions)
                    # Fold branch -> P1 wins
                    build_subtree(nid, cards, actions + (Action(0, 'f'),), depth + 1, chance_prob)
                    # Call branch -> showdown
                    build_subtree(nid, cards, actions + (Action(1, 'c'),), depth + 1, chance_prob)

            elif len(actions) == 2:
                if actions[0].id == 0 and actions[1].id == 0:
                    # Check-Check -> showdown
                    # Both contributed 1 (ante only)
                    util = terminal_utility(cards, 1, 1, None)
                    add_node(parent_id, Player.CHANCE, history, actions[-1], depth, True, util, 1.0, ())

                elif actions[0].id == 0 and actions[1].id == 1:
                    # Check-Bet -> P1 responds
                    p1_actions = (Action(0, 'f'), Action(1, 'c'))
                    nid = add_node(parent_id, Player.PLAYER_1, history, actions[-1], depth, False, None, 1.0, p1_actions)
                    # Fold -> P2 wins
                    build_subtree(nid, cards, actions + (Action(0, 'f'),), depth + 1, chance_prob)
                    # Call -> showdown
                    build_subtree(nid, cards, actions + (Action(1, 'c'),), depth + 1, chance_prob)

                elif actions[0].id == 1 and actions[1].id == 0:
                    # Bet-Fold -> P1 wins
                    # P1 contributed 2 (1 ante + 1 bet), P2 contributed 1 (ante only)
                    util = terminal_utility(cards, 2, 1, 0)
                    add_node(parent_id, Player.CHANCE, history, actions[-1], depth, True, util, 1.0, ())

                elif actions[0].id == 1 and actions[1].id == 1:
                    # Bet-Call -> showdown
                    # Both contributed 2 (1 ante + 1 bet/call)
                    util = terminal_utility(cards, 2, 2, None)
                    add_node(parent_id, Player.CHANCE, history, actions[-1], depth, True, util, 1.0, ())

            elif len(actions) == 3:
                if actions[0].id == 0 and actions[1].id == 1 and actions[2].id == 0:
                    # Check-Bet-Fold -> P2 wins
                    # P1 contributed 1 (ante), P2 contributed 2 (ante + bet)
                    util = terminal_utility(cards, 1, 2, 1)
                    add_node(parent_id, Player.CHANCE, history, actions[-1], depth, True, util, 1.0, ())

                elif actions[0].id == 0 and actions[1].id == 1 and actions[2].id == 1:
                    # Check-Bet-Call -> showdown
                    # Both contributed 2 (ante + bet/call)
                    util = terminal_utility(cards, 2, 2, None)
                    add_node(parent_id, Player.CHANCE, history, actions[-1], depth, True, util, 1.0, ())

        # Root node (chance)
        root_id = add_node(None, Player.CHANCE, ((), ()), None, 0, False, None, 1.0, ())

        # All possible card deals (6 permutations of 2 cards from 3)
        card_deals = list(permutations([JACK, QUEEN, KING], 2))
        chance_prob = 1.0 / len(card_deals)

        for cards in card_deals:
            # Chance node for this deal
            deal_history = (cards, ())
            deal_node_id = add_node(
                root_id, Player.CHANCE, deal_history, None, 1, False, None, chance_prob, ()
            )
            # Build game tree from this deal
            build_subtree(deal_node_id, cards, (), 2, chance_prob)

        # Convert to numpy arrays
        num_nodes = len(nodes)
        num_terminals = sum(1 for n in nodes if n.is_terminal)
        num_decision = sum(1 for n in nodes if not n.is_terminal and n.player in (Player.PLAYER_1, Player.PLAYER_2))

        terminal_indices = np.array([n.id for n in nodes if n.is_terminal], dtype=np.int32)
        decision_indices = np.array([n.id for n in nodes if not n.is_terminal and n.player in (Player.PLAYER_1, Player.PLAYER_2)], dtype=np.int32)

        # Terminal utilities
        terminal_utilities = np.zeros((num_terminals, 2), dtype=np.float32)
        for i, tid in enumerate(terminal_indices):
            terminal_utilities[i] = nodes[tid].utility

        # Parent IDs
        parent_ids = np.array([n.parent_id if n.parent_id is not None else -1 for n in nodes], dtype=np.int32)

        # Depths
        depths = np.array([n.depth for n in nodes], dtype=np.int32)
        max_depth = int(depths.max())

        # Players
        players = np.array([n.player.value if isinstance(n.player, Player) else n.player for n in nodes], dtype=np.int32)
        # Mark terminals as -2
        for i, n in enumerate(nodes):
            if n.is_terminal:
                players[i] = -2

        # Node to infoset mapping
        node_to_infoset = np.array([n.infoset_id if n.infoset_id is not None else -1 for n in nodes], dtype=np.int32)

        # Action from parent (using action id, -1 for root/chance outcomes)
        action_from_parent = np.zeros(num_nodes, dtype=np.int32) - 1
        for n in nodes:
            if n.action_from_parent is not None:
                action_from_parent[n.id] = n.action_from_parent.id

        # Infoset num actions
        infoset_num_actions = np.array([len(info.actions) for info in infoset_list], dtype=np.int32)

        # Chance probabilities
        chance_probs = np.array([n.chance_prob for n in nodes], dtype=np.float32)

        return GameTree(
            nodes=nodes,
            infosets=infoset_list,
            num_nodes=num_nodes,
            num_terminals=num_terminals,
            num_decision_nodes=num_decision,
            num_infosets=len(infoset_list),
            num_players=2,
            max_depth=max_depth,
            terminal_indices=terminal_indices,
            decision_indices=decision_indices,
            terminal_utilities=terminal_utilities,
            parent_ids=parent_ids,
            depths=depths,
            players=players,
            node_to_infoset=node_to_infoset,
            action_from_parent=action_from_parent,
            infoset_num_actions=infoset_num_actions,
            chance_probs=chance_probs
        )
