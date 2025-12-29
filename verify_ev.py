"""Verify EV calculations manually."""

import numpy as np
from gpu_poker_cfr.games.custom_river_toy import make_aa_kk_vs_ak_aq_game
from gpu_poker_cfr.games.cards import card_name
from gpu_poker_cfr.games.hand_eval import evaluate_7cards, HAND_NAMES

game = make_aa_kk_vs_ak_aq_game()
board = list(game.board)

print("Board:", " ".join(card_name(c) for c in board))
print()

# Count combos and EV for OOP betting AJ
print("=== EV Analysis for OOP betting AJ ===")
print()

# Find an AJ combo for OOP
oop_aj = None
for h in game.oop_range:
    h_sorted = tuple(sorted(h, reverse=True))
    c1_rank = h_sorted[0] // 4
    c2_rank = h_sorted[1] // 4
    if c1_rank == 12 and c2_rank == 9:  # A and J
        oop_aj = h_sorted
        break

if oop_aj is None:
    print("No AJ found in OOP range")
else:
    print(f"OOP hand: {card_name(oop_aj[0])}{card_name(oop_aj[1])}")

    # Find all IP hands that could face this
    oop_cards = set(oop_aj)
    board_cards = set(board)
    blocked = oop_cards | board_cards

    valid_ip_hands = []
    for ip_h in game.ip_range:
        if ip_h[0] not in blocked and ip_h[1] not in blocked:
            valid_ip_hands.append(ip_h)

    print(f"Valid IP hands facing OOP AJ: {len(valid_ip_hands)}")

    # Compute EV if OOP bets all-in and IP plays optimally
    # IP should call with straights (JJ, JT), fold with non-straights
    oop_cards_7 = list(oop_aj) + board
    oop_val = evaluate_7cards(np.array(oop_cards_7, dtype=np.int32))

    ev_sum = 0
    for ip_h in valid_ip_hands:
        ip_cards_7 = list(ip_h) + board
        ip_val = evaluate_7cards(np.array(ip_cards_7, dtype=np.int32))
        ip_type = HAND_NAMES[ip_val >> 20]

        # IP optimal response to all-in
        if ip_type == "Straight":
            # IP calls
            if ip_val > oop_val:
                ev = -150  # OOP loses (but straights are ties here)
            elif ip_val < oop_val:
                ev = 150  # OOP wins
            else:
                ev = 0  # Tie
            action = "call"
        else:
            # IP folds - OOP wins pot
            ev = 50
            action = "fold"

        ip_str = f"{card_name(ip_h[0])}{card_name(ip_h[1])}"
        print(f"  vs {ip_str} ({ip_type}): IP {action}, OOP EV = {ev}")
        ev_sum += ev

    avg_ev = ev_sum / len(valid_ip_hands)
    print(f"\nAverage EV for OOP betting AJ: {avg_ev:.2f}")
    print()

    # Compare to checking
    print("=== EV for OOP checking AJ ===")
    print("If OOP checks, IP bets, OOP calls:")
    ev_sum_check = 0
    for ip_h in valid_ip_hands:
        ip_cards_7 = list(ip_h) + board
        ip_val = evaluate_7cards(np.array(ip_cards_7, dtype=np.int32))
        ip_type = HAND_NAMES[ip_val >> 20]

        if ip_type == "Straight":
            # IP bets, OOP calls, tie
            ev = 0
        else:
            # IP checks back (doesn't bet losing hand), showdown
            if oop_val > ip_val:
                ev = 50  # OOP wins pot
            elif oop_val < ip_val:
                ev = -50  # OOP loses
            else:
                ev = 0  # Tie
        ev_sum_check += ev

    avg_ev_check = ev_sum_check / len(valid_ip_hands)
    print(f"Average EV for OOP checking AJ: {avg_ev_check:.2f}")
