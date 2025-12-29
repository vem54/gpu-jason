"""
Debug: Compute what the GPU kernel SHOULD see at Node 1.

Node 1 (IP turn after OOP check):
- Check -> Node 3 (CHANCE) -> River betting
- All-in -> Node 4 (OOP decision: Fold/Call)

We need to compare ev1[3] (Check path) vs ev1[4] (All-in path).
"""

import numpy as np
from gpu_poker_cfr.games.turn_toy_game import make_turn_toy_game, build_turn_tree
from gpu_poker_cfr.games.turn_toy_game import NODE_FOLD, NODE_SHOWDOWN, NODE_DECISION, NODE_CHANCE
from gpu_poker_cfr.games.hand_eval import evaluate_7cards
from gpu_poker_cfr.games.cards import card_name


def compute_expected_evs():
    """
    Compute what EVs the GPU should see at Node 1's children.
    Using uniform strategies (50/50) at all decision nodes.
    """
    print("=" * 60)
    print("EXPECTED EVs AT NODE 1's CHILDREN")
    print("=" * 60)

    game = make_turn_toy_game()
    nodes, child_indices = build_turn_tree(game)

    # Find a specific IP hand for detailed analysis
    # Let's use JJ (which has a straight and should want to go all-in)
    target_ip = None
    for d in range(game.num_deals):
        oop_hand, ip_hand = game.get_deal(d)
        if ip_hand[0] // 4 == 9 and ip_hand[1] // 4 == 9:  # JJ
            target_ip = (d, oop_hand, ip_hand)
            break

    if not target_ip:
        print("No JJ deal found")
        return

    d, oop_hand, ip_hand = target_ip
    print(f"\nAnalyzing deal {d}:")
    print(f"  OOP: {card_name(oop_hand[0])}{card_name(oop_hand[1])}")
    print(f"  IP: {card_name(ip_hand[0])}{card_name(ip_hand[1])}")

    valid_rivers = game.get_valid_rivers(d)
    n_rivers = len(valid_rivers)
    print(f"  Valid rivers: {n_rivers}")

    # Compute EVs assuming uniform strategies

    # === Node 3 (CHANCE) - Check path ===
    # After check-check on turn, go to river
    # Assuming check-check on river -> showdown at pot=100
    check_evs = []
    for river in valid_rivers:
        board = list(game.board) + [river]
        h0 = np.array(list(oop_hand) + board, dtype=np.int32)
        h1 = np.array(list(ip_hand) + board, dtype=np.int32)
        v0, v1 = evaluate_7cards(h0), evaluate_7cards(h1)

        # IP wins/loses 50 at showdown
        if v1 > v0:
            ev1 = 50
        elif v0 > v1:
            ev1 = -50
        else:
            ev1 = 0
        check_evs.append(ev1)

    avg_check_ev = np.mean(check_evs)
    sum_check_ev = np.sum(check_evs)
    print(f"\n--- Node 3 (CHANCE) child EV for IP ---")
    print(f"  Avg EV (check-check to showdown): {avg_check_ev:.2f}")
    print(f"  Sum EV: {sum_check_ev:.2f}")

    # === Node 4 (OOP decision) - All-in path ===
    # IP goes all-in, OOP can Fold or Call
    # Node 4 children: Node 8 (Fold), Node 9 (CHANCE -> showdown)

    # If OOP Folds (Node 8): IP wins pot = +50
    fold_ev_ip = 50

    # If OOP Calls (Node 9): showdown at pot=300
    call_evs = []
    for river in valid_rivers:
        board = list(game.board) + [river]
        h0 = np.array(list(oop_hand) + board, dtype=np.int32)
        h1 = np.array(list(ip_hand) + board, dtype=np.int32)
        v0, v1 = evaluate_7cards(h0), evaluate_7cards(h1)

        if v1 > v0:
            ev1 = 150
        elif v0 > v1:
            ev1 = -150
        else:
            ev1 = 0
        call_evs.append(ev1)

    avg_call_ev_ip = np.mean(call_evs)
    sum_call_ev = np.sum(call_evs)

    # OOP's decision at Node 4: EV of fold vs call
    # OOP fold EV = -50, OOP call EV = -avg_call_ev_ip (opposite of IP)
    # If OOP plays 50/50: EV for OOP = 0.5 * (-50) + 0.5 * avg_call_ev_oop
    # EV for IP = 0.5 * 50 + 0.5 * avg_call_ev_ip

    oop_call_ev = -avg_call_ev_ip  # OOP's EV when calling (opposite of IP)

    # At equilibrium, OOP might not play 50/50
    # But with uniform strategies (50/50):
    node4_ev_ip = 0.5 * fold_ev_ip + 0.5 * avg_call_ev_ip

    print(f"\n--- Node 4 (OOP decision) EV for IP ---")
    print(f"  If OOP Folds: IP EV = {fold_ev_ip}")
    print(f"  If OOP Calls: IP EV (avg over rivers) = {avg_call_ev_ip:.2f}")
    print(f"  If OOP Calls: IP EV (sum over rivers) = {sum_call_ev:.2f}")
    print(f"  With 50/50 OOP: IP EV = {node4_ev_ip:.2f}")

    # === Compare at Node 1 ===
    print(f"\n--- Node 1 child EVs for IP (with JJ) ---")
    print(f"  Check (Node 3): EV = {avg_check_ev:.2f}")
    print(f"  All-in (Node 4): EV = {node4_ev_ip:.2f}")
    print(f"  Difference: All-in - Check = {node4_ev_ip - avg_check_ev:.2f}")

    if node4_ev_ip > avg_check_ev:
        print(f"  -> IP with JJ should prefer All-in!")
    else:
        print(f"  -> IP with JJ should prefer Check")

    # What if the GPU is using SUMMED EVs instead of AVERAGED?
    print(f"\n--- If GPU uses SUMMED EVs (BUG HYPOTHESIS) ---")
    print(f"  Check (Node 3): Sum EV = {sum_check_ev:.2f}")
    print(f"  All-in (Node 4): Sum call EV + fold = ?")

    # If OOP plays 50/50 and we SUM:
    # The fold path gives 50 (single value)
    # The call path gives sum(call_evs) for all rivers
    # This creates a scale mismatch!

    print(f"\n  ** If bug exists: **")
    print(f"  Check path = SUM of {n_rivers} river EVs = {sum_check_ev:.2f}")
    print(f"  All-in + OOP folds = SINGLE value = {fold_ev_ip}")
    print(f"  This would make Check look {abs(sum_check_ev / fold_ev_ip):.1f}x better than All-in+Fold!")


def analyze_all_ip_hands():
    """Analyze Node 1 EVs for all IP hands."""
    print("\n" + "=" * 60)
    print("NODE 1 EVs FOR ALL IP HANDS")
    print("=" * 60)

    game = make_turn_toy_game()

    # Group deals by IP hand
    ip_hand_deals = {}
    for d in range(game.num_deals):
        oop_hand, ip_hand = game.get_deal(d)
        ip_norm = tuple(sorted(ip_hand, reverse=True))
        if ip_norm not in ip_hand_deals:
            ip_hand_deals[ip_norm] = []
        ip_hand_deals[ip_norm].append((d, oop_hand, ip_hand))

    print(f"\n{'IP Hand':<10} {'Check EV':>10} {'All-in EV':>10} {'Diff':>10} {'Pref':>8}")
    print("-" * 50)

    for ip_norm in sorted(ip_hand_deals.keys(), key=lambda h: (h[0], h[1]), reverse=True):
        deals = ip_hand_deals[ip_norm]
        ip_name = f"{card_name(ip_norm[0])}{card_name(ip_norm[1])}"

        # Average EVs over all deals with this IP hand
        check_evs = []
        allin_evs = []

        for d, oop_hand, ip_hand in deals:
            valid_rivers = game.get_valid_rivers(d)

            # Check EV (check-check to showdown)
            check_ev = 0
            allin_call_ev = 0
            for river in valid_rivers:
                board = list(game.board) + [river]
                h0 = np.array(list(oop_hand) + board, dtype=np.int32)
                h1 = np.array(list(ip_hand) + board, dtype=np.int32)
                v0, v1 = evaluate_7cards(h0), evaluate_7cards(h1)

                if v1 > v0:
                    check_ev += 50
                    allin_call_ev += 150
                elif v0 > v1:
                    check_ev -= 50
                    allin_call_ev -= 150

            check_ev /= len(valid_rivers)
            allin_call_ev /= len(valid_rivers)

            # All-in EV = 50/50 between OOP fold (+50) and OOP call
            allin_ev = 0.5 * 50 + 0.5 * allin_call_ev

            check_evs.append(check_ev)
            allin_evs.append(allin_ev)

        avg_check = np.mean(check_evs)
        avg_allin = np.mean(allin_evs)
        diff = avg_allin - avg_check
        pref = "All-in" if diff > 0 else "Check"

        print(f"{ip_name:<10} {avg_check:>10.1f} {avg_allin:>10.1f} {diff:>10.1f} {pref:>8}")


if __name__ == '__main__':
    compute_expected_evs()
    analyze_all_ip_hands()
