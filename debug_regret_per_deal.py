"""
Debug: Trace regret accumulation per deal for IP JJ at Node 1.
This helps identify if certain deals are dominating the regret accumulation.
"""

import numpy as np
import cupy as cp
from gpu_poker_cfr.games.turn_toy_game import make_turn_toy_game, build_turn_tree
from gpu_poker_cfr.games.turn_toy_game import NODE_FOLD, NODE_SHOWDOWN, NODE_DECISION, NODE_CHANCE
from gpu_poker_cfr.games.cards import card_name
from gpu_poker_cfr.games.hand_eval import evaluate_7cards


def compute_deal_regrets():
    """Compute regret contribution from each deal for IP JJ at Node 1."""
    print("=" * 60)
    print("REGRET PER DEAL FOR IP JJ AT NODE 1")
    print("=" * 60)

    game = make_turn_toy_game()

    # Find all deals where IP has JJ
    jj_deals = []
    for d in range(game.num_deals):
        oop_hand, ip_hand = game.get_deal(d)
        if ip_hand[0] // 4 == 9 and ip_hand[1] // 4 == 9:  # JJ
            jj_deals.append((d, oop_hand, ip_hand))

    print(f"\nFound {len(jj_deals)} deals where IP has JJ")

    # For each deal, compute EVs for Check and All-in paths
    print(f"\n{'OOP Hand':<10} {'Check EV':>10} {'All-in EV':>12} {'Diff':>10} {'Check Reg':>10} {'All-in Reg':>12}")
    print("-" * 70)

    total_check_regret = 0
    total_allin_regret = 0

    for d, oop_hand, ip_hand in jj_deals:
        oop_name = f"{card_name(oop_hand[0])}{card_name(oop_hand[1])}"
        valid_rivers = game.get_valid_rivers(d)
        n_rivers = len(valid_rivers)

        # Check path EV: check-check to showdown (pot = 100)
        check_ev = 0
        for river in valid_rivers:
            board = list(game.board) + [river]
            h0 = np.array(list(oop_hand) + board, dtype=np.int32)
            h1 = np.array(list(ip_hand) + board, dtype=np.int32)
            v0, v1 = evaluate_7cards(h0), evaluate_7cards(h1)

            if v1 > v0:
                check_ev += 50  # IP wins
            elif v0 > v1:
                check_ev += -50  # IP loses
        check_ev /= n_rivers  # Average over rivers

        # All-in path EV: OOP folds (+50) or calls (showdown at pot=300)
        # Using uniform 50/50 for OOP's response
        fold_ev = 50  # OOP folds, IP wins pot

        call_ev = 0
        for river in valid_rivers:
            board = list(game.board) + [river]
            h0 = np.array(list(oop_hand) + board, dtype=np.int32)
            h1 = np.array(list(ip_hand) + board, dtype=np.int32)
            v0, v1 = evaluate_7cards(h0), evaluate_7cards(h1)

            if v1 > v0:
                call_ev += 150  # IP wins big pot
            elif v0 > v1:
                call_ev += -150  # IP loses big pot
        call_ev /= n_rivers  # Average over rivers

        # OOP's decision with uniform 50/50
        allin_ev = 0.5 * fold_ev + 0.5 * call_ev

        # Regret calculation (assuming uniform strategy at Node 1)
        # node_ev = 0.5 * check_ev + 0.5 * allin_ev
        # check_regret = check_ev - node_ev = 0.5 * (check_ev - allin_ev)
        # allin_regret = allin_ev - node_ev = 0.5 * (allin_ev - check_ev)

        # With uniform OOP reach = 1.0
        node_ev = 0.5 * check_ev + 0.5 * allin_ev
        check_regret = check_ev - node_ev
        allin_regret = allin_ev - node_ev

        total_check_regret += check_regret
        total_allin_regret += allin_regret

        diff = allin_ev - check_ev
        print(f"{oop_name:<10} {check_ev:>10.1f} {allin_ev:>12.1f} {diff:>+10.1f} {check_regret:>+10.1f} {allin_regret:>+12.1f}")

    print("-" * 70)
    print(f"{'TOTAL':<10} {'':>10} {'':>12} {'':>10} {total_check_regret:>+10.1f} {total_allin_regret:>+12.1f}")

    print(f"\n--- Summary ---")
    print(f"Total Check regret: {total_check_regret:.1f}")
    print(f"Total All-in regret: {total_allin_regret:.1f}")
    print(f"Expected optimal: {'All-in' if total_allin_regret > total_check_regret else 'Check'}")

    # Now compute what happens when OOP plays optimally
    print("\n" + "=" * 60)
    print("WHEN OOP PLAYS OPTIMALLY AT NODE 4")
    print("=" * 60)

    # OOP should fold when expected showdown EV is worse than -50
    # OOP should call when expected showdown EV is better than -50

    print(f"\n{'OOP Hand':<10} {'Showdown':>10} {'OOP Action':>12} {'IP All-in EV':>14}")
    print("-" * 50)

    total_check_ev_optimal = 0
    total_allin_ev_optimal = 0

    for d, oop_hand, ip_hand in jj_deals:
        oop_name = f"{card_name(oop_hand[0])}{card_name(oop_hand[1])}"
        valid_rivers = game.get_valid_rivers(d)
        n_rivers = len(valid_rivers)

        # Check EV stays the same
        check_ev = 0
        for river in valid_rivers:
            board = list(game.board) + [river]
            h0 = np.array(list(oop_hand) + board, dtype=np.int32)
            h1 = np.array(list(ip_hand) + board, dtype=np.int32)
            v0, v1 = evaluate_7cards(h0), evaluate_7cards(h1)
            if v1 > v0:
                check_ev += 50
            elif v0 > v1:
                check_ev += -50
        check_ev /= n_rivers

        # OOP's showdown EV when calling
        oop_showdown_ev = 0  # From OOP's perspective
        for river in valid_rivers:
            board = list(game.board) + [river]
            h0 = np.array(list(oop_hand) + board, dtype=np.int32)
            h1 = np.array(list(ip_hand) + board, dtype=np.int32)
            v0, v1 = evaluate_7cards(h0), evaluate_7cards(h1)
            if v0 > v1:
                oop_showdown_ev += 150  # OOP wins
            elif v1 > v0:
                oop_showdown_ev += -150  # OOP loses
        oop_showdown_ev /= n_rivers

        # OOP folds: loses 50 (already invested)
        # OOP calls: showdown EV
        # OOP should call if showdown_ev > -50

        if oop_showdown_ev > -50:
            oop_action = "CALL"
            # IP's EV = -oop_showdown_ev (they're playing heads up)
            ip_allin_ev = -oop_showdown_ev
        else:
            oop_action = "FOLD"
            ip_allin_ev = 50  # IP wins the pot

        total_check_ev_optimal += check_ev
        total_allin_ev_optimal += ip_allin_ev

        print(f"{oop_name:<10} {oop_showdown_ev:>+10.1f} {oop_action:>12} {ip_allin_ev:>+14.1f}")

    print("-" * 50)
    avg_check = total_check_ev_optimal / len(jj_deals)
    avg_allin = total_allin_ev_optimal / len(jj_deals)
    print(f"{'AVG':<10} {'':>10} {'':>12} {'':>14}")
    print(f"Check EV: {avg_check:.1f}, All-in EV: {avg_allin:.1f}")
    print(f"Optimal for IP JJ: {'All-in' if avg_allin > avg_check else 'Check'}")


def trace_oop_adaptation():
    """Trace how OOP adapts their strategy at Node 4."""
    print("\n" + "=" * 60)
    print("OOP ADAPTATION AT NODE 4 (FACING ALL-IN)")
    print("=" * 60)

    game = make_turn_toy_game()

    # For each OOP hand, compute their EV for fold vs call
    print(f"\n{'OOP Hand':<10} {'Fold EV':>10} {'Call EV':>10} {'Optimal':>10}")
    print("-" * 45)

    oop_hands = list(set(tuple(sorted(h, reverse=True)) for h in game.oop_range))

    for oop_hand in sorted(oop_hands, key=lambda h: (h[0], h[1]), reverse=True):
        oop_name = f"{card_name(oop_hand[0])}{card_name(oop_hand[1])}"

        fold_ev = -50  # OOP loses investment

        # Compute average call EV against all IP hands
        call_evs = []
        for d in range(game.num_deals):
            deal_oop, deal_ip = game.get_deal(d)
            if tuple(sorted(deal_oop, reverse=True)) != oop_hand:
                continue

            valid_rivers = game.get_valid_rivers(d)
            deal_call_ev = 0
            for river in valid_rivers:
                board = list(game.board) + [river]
                h0 = np.array(list(deal_oop) + board, dtype=np.int32)
                h1 = np.array(list(deal_ip) + board, dtype=np.int32)
                v0, v1 = evaluate_7cards(h0), evaluate_7cards(h1)
                if v0 > v1:
                    deal_call_ev += 150
                elif v1 > v0:
                    deal_call_ev += -150
            deal_call_ev /= len(valid_rivers)
            call_evs.append(deal_call_ev)

        if call_evs:
            avg_call_ev = np.mean(call_evs)
            optimal = "CALL" if avg_call_ev > fold_ev else "FOLD"
            print(f"{oop_name:<10} {fold_ev:>+10.1f} {avg_call_ev:>+10.1f} {optimal:>10}")


if __name__ == '__main__':
    compute_deal_regrets()
    trace_oop_adaptation()
