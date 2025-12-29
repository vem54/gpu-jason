"""
Debug: Trace EV computation at turn decision nodes.
We'll add debug output to see what EVs are being used.
"""

import numpy as np
from gpu_poker_cfr.games.turn_toy_game import (
    make_turn_toy_game, build_turn_tree,
    NODE_FOLD, NODE_SHOWDOWN, NODE_DECISION, NODE_CHANCE
)
from gpu_poker_cfr.games.hand_eval import evaluate_7cards
from gpu_poker_cfr.games.cards import card_name


def trace_ev_computation():
    """
    Manually trace EV computation for OOP with AA at Node 0.

    WASM/Pio says: EV Bet = 38.21, EV Check = 40.16
    """
    print("=" * 60)
    print("TRACING EV COMPUTATION FOR OOP AA AT NODE 0")
    print("=" * 60)

    game = make_turn_toy_game()
    nodes, child_indices = build_turn_tree(game)

    # Find deals where OOP has AA
    aa_deals = []
    for d in range(game.num_deals):
        oop_hand, ip_hand = game.get_deal(d)
        if oop_hand[0] // 4 == 12 and oop_hand[1] // 4 == 12:  # Both Aces
            aa_deals.append((d, oop_hand, ip_hand))

    print(f"\nFound {len(aa_deals)} deals with OOP AA")

    # Trace EV for first AA deal
    if not aa_deals:
        print("No AA deals found!")
        return

    d, oop_hand, ip_hand = aa_deals[0]
    print(f"\nDeal {d}:")
    print(f"  OOP: {card_name(oop_hand[0])} {card_name(oop_hand[1])}")
    print(f"  IP: {card_name(ip_hand[0])} {card_name(ip_hand[1])}")

    # Get valid rivers
    valid_rivers = game.get_valid_rivers(d)
    n_rivers = len(valid_rivers)
    river_weight = 1.0 / n_rivers
    print(f"  Valid rivers: {n_rivers}")

    # Tree structure at Node 0:
    # - Check -> Node 1 (IP decision) -> Node 3 (CHANCE) or Node 4 (OOP decision)
    # - All-in -> Node 2 (IP decision: Fold/Call)

    print("\n--- Node 0 (OOP turn): Check vs All-in ---")
    print(f"Children: Check -> Node 1, All-in -> Node 2")

    # === Trace CHECK path (Node 0 -> Node 1 -> ...) ===
    print("\n=== CHECK PATH ===")
    # If OOP checks, IP acts at Node 1 (Check/All-in)
    # For now, assume IP also checks (goes to CHANCE -> River)

    # Node 1 -> Check -> Node 3 (CHANCE) -> Node 7 (River OOP decision)
    # Compute EV assuming check-through to showdown
    check_check_evs = []
    for river in valid_rivers:
        board_with_river = list(game.board) + [river]
        h0 = np.array(list(oop_hand) + board_with_river, dtype=np.int32)
        h1 = np.array(list(ip_hand) + board_with_river, dtype=np.int32)
        v0 = evaluate_7cards(h0)
        v1 = evaluate_7cards(h1)

        # Showdown at pot=100, invested_oop=50, invested_ip=50
        if v0 > v1:
            ev0 = 50  # OOP wins IP's investment
        elif v1 > v0:
            ev0 = -50  # OOP loses
        else:
            ev0 = 0  # Chop

        check_check_evs.append(ev0)

    sum_check_ev = sum(check_check_evs)
    avg_check_ev = sum_check_ev / len(check_check_evs)
    print(f"  If IP also checks (check-check to showdown):")
    print(f"    Sum of river EVs: {sum_check_ev:.2f}")
    print(f"    Avg of river EVs: {avg_check_ev:.2f}")

    # === Trace ALL-IN path (Node 0 -> Node 2: IP faces all-in) ===
    print("\n=== ALL-IN PATH ===")
    # Node 2: IP decision (Fold/Call)
    # - If IP folds: OOP wins pot (EV = +50 for OOP)
    # - If IP calls: Chance -> Showdown at pot=300

    print("  Node 2 (IP facing all-in):")
    print("    - Fold: OOP EV = +50 (OOP wins IP's 50)")

    # If IP calls: all-in showdown
    allin_call_evs = []
    for river in valid_rivers:
        board_with_river = list(game.board) + [river]
        h0 = np.array(list(oop_hand) + board_with_river, dtype=np.int32)
        h1 = np.array(list(ip_hand) + board_with_river, dtype=np.int32)
        v0 = evaluate_7cards(h0)
        v1 = evaluate_7cards(h1)

        # Showdown at pot=300, invested_oop=150, invested_ip=150
        if v0 > v1:
            ev0 = 150  # OOP wins IP's investment
        elif v1 > v0:
            ev0 = -150  # OOP loses
        else:
            ev0 = 0  # Chop

        allin_call_evs.append(ev0)

    sum_call_ev = sum(allin_call_evs)
    avg_call_ev = sum_call_ev / len(allin_call_evs)
    print(f"    - Call: OOP EV = {avg_call_ev:.2f} (averaged over rivers)")
    print(f"      Sum: {sum_call_ev:.2f}, Avg: {avg_call_ev:.2f}")

    # === Expected EVs based on equilibrium strategies ===
    print("\n=== EXPECTED VS WASM/PIO ===")
    print(f"WASM/Pio for OOP AA: EV Bet = 38.21, EV Check = 40.16")
    print(f"Our check-check EV: {avg_check_ev:.2f}")
    print(f"Our all-in (if IP folds 100%): +50")
    print(f"Our all-in (if IP calls 100%): {avg_call_ev:.2f}")

    # If IP mixes at Node 2, the all-in EV would be a mix
    # WASM says EV Bet = 38.21, which suggests IP calls quite often

    # For the actual tree:
    # - EV Check should account for IP's strategy at Node 1
    # - EV All-in should account for IP's strategy at Node 2

    print("\n--- Computing what our solver would see ---")

    # At Node 0, the child EVs are:
    # - child[0] (Check) = EV at Node 1
    # - child[1] (All-in) = EV at Node 2

    # EV at Node 1 depends on IP's strategy (check or all-in)
    # EV at Node 2 depends on IP's strategy (fold or call)

    # For a simplified analysis, assume uniform strategies:
    # IP at Node 1: 50% check, 50% all-in
    # IP at Node 2: 50% fold, 50% call

    # EV at Node 1 (IP 50/50):
    ev_node1_check = avg_check_ev  # If IP checks
    # If IP all-ins: OOP faces decision at Node 4
    # Assume OOP folds: EV = -50, calls: EV = ?
    # (This gets complicated - need to trace further)

    # Simplified: assume both check through
    print(f"Simplified EV at Node 1 (assume check-through): {avg_check_ev:.2f}")

    # EV at Node 2 (IP 50/50 fold/call):
    ev_node2 = 0.5 * 50 + 0.5 * avg_call_ev
    print(f"Simplified EV at Node 2 (IP 50/50): {ev_node2:.2f}")

    print("\n" + "=" * 60)
    print("KEY INSIGHT: If these EVs are ~40 range, our solver is correct.")
    print("If Check EV is ~1800 (summed, not averaged), there's a bug.")
    print("=" * 60)


if __name__ == '__main__':
    trace_ev_computation()
