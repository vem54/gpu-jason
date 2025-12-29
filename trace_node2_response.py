"""
Trace IP's response at Node 2 (facing OOP all-in).

At Node 2, IP must decide Fold or Call facing OOP's 100 all-in.
IP's decision depends on their hand's equity vs OOP's range.
"""

import numpy as np
from gpu_poker_cfr.games.turn_toy_game import make_turn_toy_game
from gpu_poker_cfr.games.hand_eval import evaluate_7cards
from gpu_poker_cfr.games.cards import card_name


def trace_node2_response():
    print("=" * 60)
    print("IP RESPONSE TO OOP ALL-IN (Node 2)")
    print("=" * 60)

    game = make_turn_toy_game()

    # For each IP hand, compute EV of Fold vs Call against OOP's range
    ip_hands = list(set(tuple(sorted(h, reverse=True)) for h in game.ip_range))
    oop_hands = list(set(tuple(sorted(h, reverse=True)) for h in game.oop_range))

    print(f"\nIP hands: {len(ip_hands)}")
    print(f"OOP hands (range): {len(oop_hands)}")

    print("\n--- IP's decision at Node 2 (facing OOP all-in) ---")

    for ip_hand in sorted(ip_hands, key=lambda h: (h[0], h[1]), reverse=True):
        ip_name = f"{card_name(ip_hand[0])}{card_name(ip_hand[1])}"

        # EV of Fold: IP loses their 50 invested = EV = -50... wait, that's not right
        # At Node 2, pot = 200 (100 original + 100 OOP all-in)
        # IP has invested 50 so far
        # If IP folds: IP loses their 50, EV = -50

        # If IP calls (adds 100):
        # Pot = 300, IP invested 150, OOP invested 150
        # EV = avg over rivers of (win_prob * 150 - lose_prob * 150)

        # But we need to weight by valid deals for this IP hand vs OOP's range
        call_evs = []
        n_deals = 0

        for oop_hand in oop_hands:
            # Check if hands conflict
            if set(oop_hand) & set(ip_hand):
                continue
            if set(oop_hand) & set(game.board):
                continue
            if set(ip_hand) & set(game.board):
                continue

            # Find valid rivers
            blocked = set(oop_hand) | set(ip_hand) | set(game.board)
            valid_rivers = [r for r in range(52) if r not in blocked]

            wins = losses = ties = 0
            for river in valid_rivers:
                board = list(game.board) + [river]
                h0 = np.array(list(oop_hand) + board, dtype=np.int32)
                h1 = np.array(list(ip_hand) + board, dtype=np.int32)
                v0, v1 = evaluate_7cards(h0), evaluate_7cards(h1)
                if v1 > v0:
                    wins += 1  # IP wins
                elif v0 > v1:
                    losses += 1
                else:
                    ties += 1

            # EV of calling
            call_ev = (wins * 150 - losses * 150) / len(valid_rivers)
            call_evs.append(call_ev)
            n_deals += 1

        if n_deals == 0:
            continue

        avg_call_ev = np.mean(call_evs)
        fold_ev = -50  # IP loses their investment

        # IP should call if call_ev > fold_ev
        should_call = avg_call_ev > fold_ev

        print(f"{ip_name}: Fold EV = {fold_ev}, Call EV = {avg_call_ev:.1f} "
              f"-> {'CALL' if should_call else 'FOLD'} ({n_deals} deals)")


def trace_node4_response():
    """
    Trace OOP's response at Node 4 (facing IP all-in after OOP check).

    After OOP Check, IP all-in. OOP must decide Fold or Call.
    """
    print("\n" + "=" * 60)
    print("OOP RESPONSE TO IP ALL-IN (Node 4)")
    print("=" * 60)

    game = make_turn_toy_game()

    oop_hands = list(set(tuple(sorted(h, reverse=True)) for h in game.oop_range))
    ip_hands = list(set(tuple(sorted(h, reverse=True)) for h in game.ip_range))

    print(f"\nOOP hands: {len(oop_hands)}")

    print("\n--- OOP's decision at Node 4 (facing IP all-in) ---")

    for oop_hand in sorted(oop_hands, key=lambda h: (h[0], h[1]), reverse=True):
        oop_name = f"{card_name(oop_hand[0])}{card_name(oop_hand[1])}"

        call_evs = []
        n_deals = 0

        for ip_hand in ip_hands:
            if set(oop_hand) & set(ip_hand):
                continue
            if set(oop_hand) & set(game.board):
                continue
            if set(ip_hand) & set(game.board):
                continue

            blocked = set(oop_hand) | set(ip_hand) | set(game.board)
            valid_rivers = [r for r in range(52) if r not in blocked]

            wins = losses = 0
            for river in valid_rivers:
                board = list(game.board) + [river]
                h0 = np.array(list(oop_hand) + board, dtype=np.int32)
                h1 = np.array(list(ip_hand) + board, dtype=np.int32)
                v0, v1 = evaluate_7cards(h0), evaluate_7cards(h1)
                if v0 > v1:
                    wins += 1
                elif v1 > v0:
                    losses += 1

            call_ev = (wins * 150 - losses * 150) / len(valid_rivers)
            call_evs.append(call_ev)
            n_deals += 1

        if n_deals == 0:
            continue

        avg_call_ev = np.mean(call_evs)
        fold_ev = -50  # OOP loses their investment

        should_call = avg_call_ev > fold_ev
        print(f"{oop_name}: Fold EV = {fold_ev}, Call EV = {avg_call_ev:.1f} "
              f"-> {'CALL' if should_call else 'FOLD'} ({n_deals} deals)")


if __name__ == '__main__':
    trace_node2_response()
    trace_node4_response()
