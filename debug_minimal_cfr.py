"""
Minimal CFR test with ONE deal to trace exact EV computation.
"""

import numpy as np
from gpu_poker_cfr.games.turn_toy_game import (
    TurnGame, build_turn_tree, print_tree_structure,
    NODE_FOLD, NODE_SHOWDOWN, NODE_DECISION, NODE_CHANCE
)
from gpu_poker_cfr.games.hand_eval import evaluate_7cards
from gpu_poker_cfr.games.cards import make_card, card_name
from gpu_poker_cfr.games.cards import RANK_A, RANK_K, SPADES, HEARTS, CLUBS, DIAMONDS


def create_minimal_game():
    """Create a game with just AA vs KK."""
    # Board: As Ks Qs Th
    board = [
        make_card(RANK_A, SPADES),
        make_card(RANK_K, SPADES),
        make_card(12, DIAMONDS),  # Qs equivalent - actually Qd
        make_card(8, HEARTS),     # Th
    ]
    # Wait, let me use the actual cards from our game
    # Board: As=51, Ks=47, Qs=43, Th=34

    # Actually, let's just use two specific hands
    # OOP: Ac Ad (50, 49)
    # IP: Kc Kd (46, 45)

    oop_range = [(50, 49)]  # Ac Ad
    ip_range = [(46, 45)]   # Kc Kd

    class MinimalGame:
        def __init__(self):
            self.board = [51, 47, 43, 34]  # As Ks Qs Th
            self.oop_range = oop_range
            self.ip_range = ip_range
            self.pot_size = 100
            self.stack_size = 100
            self.deals = [(oop_range[0], ip_range[0])]
            self.num_deals = 1

            # Compute possible rivers
            all_cards = set(range(52))
            board_set = set(self.board)
            deal_cards = set(oop_range[0]) | set(ip_range[0])
            self.possible_rivers = sorted(all_cards - board_set)
            self.num_rivers = len(self.possible_rivers)

        def get_deal(self, deal_idx):
            return self.deals[deal_idx]

        def get_valid_rivers(self, deal_idx):
            oop_hand, ip_hand = self.deals[deal_idx]
            blocked = set(oop_hand) | set(ip_hand)
            return [r for r in self.possible_rivers if r not in blocked]

    return MinimalGame()


def manual_cfr_iteration(game, strategy=None):
    """
    Run one CFR iteration MANUALLY to trace exact computation.
    """
    nodes, child_indices = build_turn_tree_minimal(game)

    if strategy is None:
        # Uniform strategy
        strategy = {i: [0.5, 0.5] for i in range(len(nodes)) if nodes[i]['type'] == NODE_DECISION}

    print("\n--- Manual CFR Iteration ---")
    print(f"Strategy: {strategy}")

    # Get valid rivers
    deal_idx = 0
    oop_hand, ip_hand = game.get_deal(deal_idx)
    valid_rivers = game.get_valid_rivers(deal_idx)
    n_rivers = len(valid_rivers)
    river_weight = 1.0 / n_rivers

    print(f"\nDeal 0: OOP={card_name(oop_hand[0])}{card_name(oop_hand[1])}, IP={card_name(ip_hand[0])}{card_name(ip_hand[1])}")
    print(f"Valid rivers: {n_rivers}")

    # Precompute hand values for all rivers
    hand_values = {}
    for river in valid_rivers:
        board_with_river = list(game.board) + [river]
        h0 = np.array(list(oop_hand) + board_with_river, dtype=np.int32)
        h1 = np.array(list(ip_hand) + board_with_river, dtype=np.int32)
        hand_values[river] = (evaluate_7cards(h0), evaluate_7cards(h1))

    # === Compute EVs bottom-up ===
    # We'll compute EV for each (node, river) combination for river nodes
    # and (node,) for turn nodes

    ev0 = {}  # ev0[(node_idx, river)] for river nodes, ev0[node_idx] for turn
    ev1 = {}

    # Process all rivers
    chance_ev0_sum = {3: 0.0, 6: 0.0, 9: 0.0}  # For chance nodes
    chance_ev1_sum = {3: 0.0, 6: 0.0, 9: 0.0}

    for river in valid_rivers:
        v0, v1 = hand_values[river]

        # River terminals
        # Node 15: SHOWDOWN (check-check-check-check)
        if v0 > v1:
            ev0[(15, river)] = 50
            ev1[(15, river)] = -50
        elif v1 > v0:
            ev0[(15, river)] = -50
            ev1[(15, river)] = 50
        else:
            ev0[(15, river)] = 0
            ev1[(15, river)] = 0

        # ... (simplified - just trace the check-check path through Node 3)

        # Accumulate for chance node 3
        # (This is simplified - we'd need to trace all paths)

    # Let me just print the expected values based on the tree structure
    print("\n--- Tree Structure ---")
    print_tree_structure(nodes, child_indices)


def build_turn_tree_minimal(game):
    """Wrapper for build_turn_tree."""
    # Use the existing build_turn_tree
    return build_turn_tree(game)


def trace_gpu_ev():
    """
    Trace what the GPU kernel computes.
    """
    print("=" * 60)
    print("TRACING GPU EV COMPUTATION")
    print("=" * 60)

    game = create_minimal_game()
    print(f"\nMinimal game:")
    print(f"  OOP: {card_name(game.oop_range[0][0])}{card_name(game.oop_range[0][1])}")
    print(f"  IP: {card_name(game.ip_range[0][0])}{card_name(game.ip_range[0][1])}")
    print(f"  Valid rivers: {len(game.get_valid_rivers(0))}")

    # Build tree
    nodes, child_indices = build_turn_tree(game)
    print_tree_structure(nodes, child_indices)

    # Compute expected EVs manually
    print("\n--- Manual EV Computation (uniform strategies) ---")

    # For OOP at Node 0:
    # - Check leads to Node 1 (IP decision)
    # - All-in leads to Node 2 (IP faces all-in)

    # Simplest path: Check-Check-Check-Check -> Showdown
    valid_rivers = game.get_valid_rivers(0)
    oop_hand, ip_hand = game.get_deal(0)

    wins = 0
    losses = 0
    ties = 0
    for river in valid_rivers:
        board = list(game.board) + [river]
        h0 = np.array(list(oop_hand) + board, dtype=np.int32)
        h1 = np.array(list(ip_hand) + board, dtype=np.int32)
        v0 = evaluate_7cards(h0)
        v1 = evaluate_7cards(h1)
        if v0 > v1:
            wins += 1
        elif v1 > v0:
            losses += 1
        else:
            ties += 1

    print(f"\nShowdown results (AA vs KK):")
    print(f"  AA wins: {wins}/{len(valid_rivers)} = {wins/len(valid_rivers)*100:.1f}%")
    print(f"  KK wins: {losses}/{len(valid_rivers)} = {losses/len(valid_rivers)*100:.1f}%")
    print(f"  Ties: {ties}/{len(valid_rivers)} = {ties/len(valid_rivers)*100:.1f}%")

    avg_ev_oop = (wins * 50 - losses * 50) / len(valid_rivers)
    print(f"\nCheck-check EV for OOP: {avg_ev_oop:.2f}")

    # All-in path: OOP all-in, IP folds -> OOP wins 50
    #              OOP all-in, IP calls -> Showdown at pot 300
    avg_allin_call_ev = (wins * 150 - losses * 150) / len(valid_rivers)
    print(f"All-in (IP calls) EV for OOP: {avg_allin_call_ev:.2f}")

    # Expected WASM values: Bet EV = 38.21, Check EV = 40.16
    print(f"\nWASM/Pio says: Bet EV = 38.21, Check EV = 40.16")
    print(f"Note: These are at equilibrium, accounting for opponent's responses")


if __name__ == '__main__':
    trace_gpu_ev()
