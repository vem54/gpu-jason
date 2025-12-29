"""
Leduc Poker implementation.

Leduc Poker is a simplified poker game larger than Kuhn:
- 6-card deck: 2 Jacks (J), 2 Queens (Q), 2 Kings (K)
- Each player antes 1 chip
- Each player is dealt one private card
- Round 1: betting round (max 2 raises)
- Community card is dealt
- Round 2: betting round (max 2 raises)
- Showdown: pair (private matches community) beats high card

Betting structure:
- Round 1: bet/raise = 2 chips
- Round 2: bet/raise = 4 chips
- Max 2 raises per round

Game tree has ~5,000+ nodes, ~1,000 infosets.
"""

from typing import Dict, List, Tuple, Optional
from itertools import permutations, combinations
import numpy as np

from .base import Game, GameTree, Node, Infoset, Action, Player


# Card values
JACK = 0
QUEEN = 1
KING = 2
CARD_NAMES = {JACK: 'J', QUEEN: 'Q', KING: 'K'}

# Betting actions
FOLD = Action(id=0, name='f')
CALL = Action(id=1, name='c')  # Also used for check
BET = Action(id=2, name='b')   # Also used for raise

# Game constants
ANTE = 1
ROUND1_BET = 2
ROUND2_BET = 4
MAX_RAISES = 2


class LeducPoker(Game):
    """Leduc Poker game implementation."""

    @property
    def name(self) -> str:
        return "leduc_poker"

    @property
    def num_players(self) -> int:
        return 2

    def infoset_key(self, player: Player, history: Tuple) -> str:
        """
        Information set key for a player given history.

        Player knows: their private card, community card (if dealt), betting history
        Player doesn't know: opponent's private card
        """
        private_cards, community_card, actions, round_num = history
        my_card = private_cards[player.value]

        # Build key: private_card:community_card:action_history
        key_parts = [CARD_NAMES[my_card]]

        if community_card is not None:
            key_parts.append(CARD_NAMES[community_card])
        else:
            key_parts.append('?')

        action_str = ''.join(a.name for a in actions)
        key_parts.append(action_str)

        return ':'.join(key_parts)

    def build_tree(self) -> GameTree:
        """Build the complete Leduc Poker game tree."""
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

        def hand_rank(private_card: int, community_card: int) -> int:
            """
            Compute hand rank. Higher is better.
            Pair (private == community) beats high card.
            """
            if private_card == community_card:
                # Pair: rank 100 + card value
                return 100 + private_card
            else:
                # High card: just card value
                return private_card

        def terminal_utility(
            private_cards: Tuple[int, int],
            community_card: int,
            p1_contrib: int,
            p2_contrib: int,
            folded_player: Optional[int]
        ) -> Tuple[float, float]:
            """Calculate terminal utilities."""
            if folded_player is not None:
                # Someone folded
                if folded_player == 0:
                    # P1 folded, P2 wins P1's contribution
                    return (float(-p1_contrib), float(p1_contrib))
                else:
                    # P2 folded, P1 wins P2's contribution
                    return (float(p2_contrib), float(-p2_contrib))
            else:
                # Showdown
                p1_rank = hand_rank(private_cards[0], community_card)
                p2_rank = hand_rank(private_cards[1], community_card)

                if p1_rank > p2_rank:
                    return (float(p2_contrib), float(-p2_contrib))
                elif p2_rank > p1_rank:
                    return (float(-p1_contrib), float(p1_contrib))
                else:
                    # Tie (can happen with high card)
                    return (0.0, 0.0)

        def get_available_actions(num_raises_this_round: int, facing_bet: bool) -> Tuple[Action, ...]:
            """Get available actions based on betting state."""
            if facing_bet:
                if num_raises_this_round < MAX_RAISES:
                    return (FOLD, CALL, BET)  # fold, call, raise
                else:
                    return (FOLD, CALL)  # fold, call (no more raises)
            else:
                if num_raises_this_round < MAX_RAISES:
                    return (CALL, BET)  # check, bet
                else:
                    return (CALL,)  # check only

        def build_betting_round(
            parent_id: int,
            private_cards: Tuple[int, int],
            community_card: Optional[int],
            actions: Tuple[Action, ...],
            depth: int,
            chance_prob: float,
            round_num: int,
            p1_contrib: int,
            p2_contrib: int,
            to_act: int,  # 0 = P1, 1 = P2
            facing_bet: bool,
            num_raises: int,
            bet_size: int
        ):
            """Recursively build betting round."""
            history = (private_cards, community_card, actions, round_num)
            player = Player.PLAYER_1 if to_act == 0 else Player.PLAYER_2

            available = get_available_actions(num_raises, facing_bet)

            nid = add_node(
                parent_id, player, history,
                actions[-1] if actions else None,
                depth, False, None, 1.0, available
            )

            for action in available:
                new_actions = actions + (action,)

                if action.id == 0:  # FOLD
                    # Current player folds
                    util = terminal_utility(
                        private_cards, community_card if community_card is not None else private_cards[0],
                        p1_contrib, p2_contrib, to_act
                    )
                    new_history = (private_cards, community_card, new_actions, round_num)
                    add_node(nid, Player.CHANCE, new_history, action, depth + 1, True, util, 1.0, ())

                elif action.id == 1:  # CALL/CHECK
                    if facing_bet:
                        # Calling a bet
                        if to_act == 0:
                            new_p1 = p2_contrib  # Match opponent
                        else:
                            new_p2 = p1_contrib

                        new_p1_contrib = p1_contrib if to_act == 1 else p2_contrib
                        new_p2_contrib = p2_contrib if to_act == 0 else p1_contrib

                        if round_num == 1:
                            # End of round 1, deal community card
                            build_community_deal(
                                nid, private_cards, new_actions, depth + 1,
                                chance_prob, new_p1_contrib, new_p2_contrib
                            )
                        else:
                            # End of round 2, showdown
                            util = terminal_utility(
                                private_cards, community_card,
                                new_p1_contrib, new_p2_contrib, None
                            )
                            new_history = (private_cards, community_card, new_actions, round_num)
                            add_node(nid, Player.CHANCE, new_history, action, depth + 1, True, util, 1.0, ())
                    else:
                        # Checking
                        if to_act == 1:  # P2 checks after P1 checked
                            if round_num == 1:
                                # End of round 1
                                build_community_deal(
                                    nid, private_cards, new_actions, depth + 1,
                                    chance_prob, p1_contrib, p2_contrib
                                )
                            else:
                                # End of round 2, showdown
                                util = terminal_utility(
                                    private_cards, community_card,
                                    p1_contrib, p2_contrib, None
                                )
                                new_history = (private_cards, community_card, new_actions, round_num)
                                add_node(nid, Player.CHANCE, new_history, action, depth + 1, True, util, 1.0, ())
                        else:
                            # P1 checked, P2 to act
                            build_betting_round(
                                nid, private_cards, community_card, new_actions,
                                depth + 1, chance_prob, round_num,
                                p1_contrib, p2_contrib,
                                1, False, num_raises, bet_size
                            )

                elif action.id == 2:  # BET/RAISE
                    if to_act == 0:
                        new_p1_contrib = p1_contrib + bet_size
                        if facing_bet:
                            new_p1_contrib = p2_contrib + bet_size  # Match + raise
                        new_p2_contrib = p2_contrib
                    else:
                        new_p2_contrib = p2_contrib + bet_size
                        if facing_bet:
                            new_p2_contrib = p1_contrib + bet_size  # Match + raise
                        new_p1_contrib = p1_contrib

                    build_betting_round(
                        nid, private_cards, community_card, new_actions,
                        depth + 1, chance_prob, round_num,
                        new_p1_contrib, new_p2_contrib,
                        1 - to_act, True, num_raises + 1, bet_size
                    )

        def build_community_deal(
            parent_id: int,
            private_cards: Tuple[int, int],
            actions: Tuple[Action, ...],
            depth: int,
            chance_prob: float,
            p1_contrib: int,
            p2_contrib: int
        ):
            """Deal community card and start round 2."""
            # Possible community cards (cards not already dealt)
            all_cards = [JACK, JACK, QUEEN, QUEEN, KING, KING]
            for card in private_cards:
                all_cards.remove(card)

            # Count remaining cards
            card_counts = {}
            for card in all_cards:
                card_counts[card] = card_counts.get(card, 0) + 1

            total_cards = len(all_cards)

            for card, count in card_counts.items():
                card_prob = count / total_cards

                # Chance node for this community card
                history = (private_cards, card, actions, 2)
                deal_node_id = add_node(
                    parent_id, Player.CHANCE, history, None,
                    depth, False, None, card_prob, ()
                )

                # Start round 2 betting
                build_betting_round(
                    deal_node_id, private_cards, card, (),
                    depth + 1, chance_prob * card_prob, 2,
                    p1_contrib, p2_contrib,
                    0, False, 0, ROUND2_BET
                )

        def build_private_deal(
            parent_id: int,
            p1_card: int,
            p2_card: int,
            depth: int,
            chance_prob: float
        ):
            """Deal private cards and start round 1."""
            private_cards = (p1_card, p2_card)
            history = (private_cards, None, (), 1)

            # Initial contributions (antes)
            p1_contrib = ANTE
            p2_contrib = ANTE

            # Chance node for this deal
            deal_node_id = add_node(
                parent_id, Player.CHANCE, history, None,
                depth, False, None, chance_prob, ()
            )

            # Start round 1 betting
            build_betting_round(
                deal_node_id, private_cards, None, (),
                depth + 1, chance_prob, 1,
                p1_contrib, p2_contrib,
                0, False, 0, ROUND1_BET
            )

        # Root node (chance)
        root_id = add_node(None, Player.CHANCE, ((), None, (), 0), None, 0, False, None, 1.0, ())

        # All possible private card deals
        # 6 cards, 2 to each player = 6 * 5 / 2 = 15 ordered pairs (but we use ordered)
        deck = [JACK, JACK, QUEEN, QUEEN, KING, KING]
        deals = []
        for i, c1 in enumerate(deck):
            for j, c2 in enumerate(deck):
                if i != j:
                    deals.append((c1, c2))

        # Probability of each deal
        chance_prob = 1.0 / len(deals)

        for p1_card, p2_card in deals:
            build_private_deal(root_id, p1_card, p2_card, 1, chance_prob)

        # Build arrays
        num_nodes = len(nodes)
        num_terminals = sum(1 for n in nodes if n.is_terminal)
        num_decision = sum(1 for n in nodes if not n.is_terminal and n.player in (Player.PLAYER_1, Player.PLAYER_2))

        terminal_indices = np.array([n.id for n in nodes if n.is_terminal], dtype=np.int32)
        decision_indices = np.array([n.id for n in nodes if not n.is_terminal and n.player in (Player.PLAYER_1, Player.PLAYER_2)], dtype=np.int32)

        terminal_utilities = np.zeros((num_terminals, 2), dtype=np.float32)
        for i, tid in enumerate(terminal_indices):
            terminal_utilities[i] = nodes[tid].utility

        parent_ids = np.array([n.parent_id if n.parent_id is not None else -1 for n in nodes], dtype=np.int32)
        depths = np.array([n.depth for n in nodes], dtype=np.int32)
        max_depth = int(depths.max())

        players = np.array([n.player.value if isinstance(n.player, Player) else n.player for n in nodes], dtype=np.int32)
        for i, n in enumerate(nodes):
            if n.is_terminal:
                players[i] = -2

        node_to_infoset = np.array([n.infoset_id if n.infoset_id is not None else -1 for n in nodes], dtype=np.int32)

        action_from_parent = np.zeros(num_nodes, dtype=np.int32) - 1
        for n in nodes:
            if n.action_from_parent is not None:
                action_from_parent[n.id] = n.action_from_parent.id

        infoset_num_actions = np.array([len(info.actions) for info in infoset_list], dtype=np.int32)
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
