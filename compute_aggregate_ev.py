"""
Compute aggregate EV for OOP AA across all IP hands.
WASM says: Bet EV = 38.21, Check EV = 40.16
"""

import numpy as np
from gpu_poker_cfr.games.turn_toy_game import make_turn_toy_game
from gpu_poker_cfr.games.hand_eval import evaluate_7cards
from gpu_poker_cfr.games.cards import card_name


def compute_aggregate_ev():
    print("=" * 60)
    print("COMPUTING AGGREGATE EV FOR OOP AA")
    print("=" * 60)

    game = make_turn_toy_game()

    # Find all deals where OOP has AA
    aa_deals = []
    for d in range(game.num_deals):
        oop_hand, ip_hand = game.get_deal(d)
        # Check if OOP has pocket aces
        if oop_hand[0] // 4 == 12 and oop_hand[1] // 4 == 12:
            aa_deals.append((d, oop_hand, ip_hand))

    print(f"\nFound {len(aa_deals)} deals with OOP AA")

    # For each AA deal, compute EV of check-check and all-in paths
    check_evs = []
    allin_fold_evs = []
    allin_call_evs = []

    for d, oop_hand, ip_hand in aa_deals:
        valid_rivers = game.get_valid_rivers(d)

        # Compute showdown results at pot=100 (check-check)
        check_wins = check_losses = check_ties = 0
        for river in valid_rivers:
            board = list(game.board) + [river]
            h0 = np.array(list(oop_hand) + board, dtype=np.int32)
            h1 = np.array(list(ip_hand) + board, dtype=np.int32)
            v0, v1 = evaluate_7cards(h0), evaluate_7cards(h1)
            if v0 > v1:
                check_wins += 1
            elif v1 > v0:
                check_losses += 1
            else:
                check_ties += 1

        check_ev = (check_wins * 50 - check_losses * 50) / len(valid_rivers)
        check_evs.append(check_ev)

        # All-in path: if IP folds, OOP wins 50
        allin_fold_evs.append(50)

        # All-in path: if IP calls, showdown at pot=300
        allin_call_ev = (check_wins * 150 - check_losses * 150) / len(valid_rivers)
        allin_call_evs.append(allin_call_ev)

    # Average across all AA deals
    avg_check_ev = np.mean(check_evs)
    avg_allin_fold_ev = np.mean(allin_fold_evs)
    avg_allin_call_ev = np.mean(allin_call_evs)

    print(f"\nAggregate EVs for OOP AA (averaged over {len(aa_deals)} deals):")
    print(f"  Check-check EV: {avg_check_ev:.2f}")
    print(f"  All-in (IP folds): {avg_allin_fold_ev:.2f}")
    print(f"  All-in (IP calls): {avg_allin_call_ev:.2f}")

    # At equilibrium, IP mixes between fold and call
    # The expected Bet EV depends on IP's equilibrium strategy
    print(f"\nWASM/Pio says: Bet EV = 38.21, Check EV = 40.16")

    # Work backwards to find what IP's fold frequency would be
    # Bet_EV = fold_prob * 50 + call_prob * allin_call_ev
    # 38.21 = fold_prob * 50 + (1 - fold_prob) * avg_allin_call_ev

    if avg_allin_call_ev != 50:
        fold_prob = (avg_allin_call_ev - 38.21) / (avg_allin_call_ev - 50)
        print(f"\nImplied IP fold probability for Bet EV = 38.21:")
        print(f"  fold_prob = {fold_prob * 100:.1f}%")

    # Also print per-IP-hand breakdown
    print("\n--- Per IP hand breakdown ---")
    ip_hand_evs = {}
    for d, oop_hand, ip_hand in aa_deals:
        ip_norm = tuple(sorted(ip_hand, reverse=True))
        ip_name = f"{card_name(ip_norm[0])}{card_name(ip_norm[1])}"

        if ip_name not in ip_hand_evs:
            ip_hand_evs[ip_name] = {'check': [], 'allin_call': []}

        valid_rivers = game.get_valid_rivers(d)
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

        check_ev = (wins * 50 - losses * 50) / len(valid_rivers)
        allin_ev = (wins * 150 - losses * 150) / len(valid_rivers)
        ip_hand_evs[ip_name]['check'].append(check_ev)
        ip_hand_evs[ip_name]['allin_call'].append(allin_ev)

    for ip_name in sorted(ip_hand_evs.keys()):
        evs = ip_hand_evs[ip_name]
        avg_check = np.mean(evs['check'])
        avg_allin = np.mean(evs['allin_call'])
        count = len(evs['check'])
        print(f"  vs {ip_name}: Check={avg_check:.1f}, All-in(call)={avg_allin:.1f} ({count} combos)")


if __name__ == '__main__':
    compute_aggregate_ev()
