"""Compute aggregated strategy weighted by deals (proper card removal)."""

from gpu_poker_cfr.games.custom_river_toy import make_aa_kk_vs_ak_aq_game
from gpu_poker_cfr.games.cards import card_name
from gpu_poker_cfr.solvers.gpu_custom_river import GPUCustomRiverCFR

game = make_aa_kk_vs_ak_aq_game()
solver = GPUCustomRiverCFR(game)

# Run solver
solver.solve(10000)

# Pre-compute deal counts per hand (for proper card removal weighting)
oop_hand_deals = {}
ip_hand_deals = {}
for d in range(game.num_deals):
    oop_hand, ip_hand = game.get_deal(d)
    oop_hand = tuple(sorted(oop_hand, reverse=True))
    ip_hand = tuple(sorted(ip_hand, reverse=True))
    oop_hand_deals[oop_hand] = oop_hand_deals.get(oop_hand, 0) + 1
    ip_hand_deals[ip_hand] = ip_hand_deals.get(ip_hand, 0) + 1

print("=" * 60)
print("AGGREGATED STRATEGY (weighted by deals - proper card removal)")
print("=" * 60)

# OOP at root
print("\n--- OOP at Root ---")
total_check = 0
total_bet = 0
total_deals = 0

hand_groups = {}
for hand in solver.oop_hands:
    deals = oop_hand_deals.get(hand, 0)

    c1_rank = hand[0] // 4
    c2_rank = hand[1] // 4

    if c1_rank == c2_rank:
        hand_type = f"{['2','3','4','5','6','7','8','9','T','J','Q','K','A'][c1_rank]}{['2','3','4','5','6','7','8','9','T','J','Q','K','A'][c1_rank]}"
    else:
        r1 = ['2','3','4','5','6','7','8','9','T','J','Q','K','A'][max(c1_rank, c2_rank)]
        r2 = ['2','3','4','5','6','7','8','9','T','J','Q','K','A'][min(c1_rank, c2_rank)]
        hand_type = f"{r1}{r2}"

    if hand_type not in hand_groups:
        hand_groups[hand_type] = {'check': 0, 'bet': 0, 'deals': 0}

    strat = solver.get_strategy_for_hand(0, hand, action_id=0)
    check = strat.get('Check', 0)
    bet = strat.get('All-in', 0)

    hand_groups[hand_type]['check'] += check * deals
    hand_groups[hand_type]['bet'] += bet * deals
    hand_groups[hand_type]['deals'] += deals

    total_check += check * deals
    total_bet += bet * deals
    total_deals += deals

print(f"{'Hand':<8} | {'Deals':<6} | {'Check':<8} | {'All-in':<8}")
print("-" * 45)
for ht in sorted(hand_groups.keys(), key=lambda x: -hand_groups[x]['deals']):
    g = hand_groups[ht]
    avg_check = g['check'] / g['deals'] * 100
    avg_bet = g['bet'] / g['deals'] * 100
    print(f"{ht:<8} | {g['deals']:<6} | {avg_check:6.1f}%  | {avg_bet:6.1f}%")

print("-" * 45)
agg_check = total_check / total_deals * 100
agg_bet = total_bet / total_deals * 100
print(f"{'TOTAL':<8} | {total_deals:<6} | {agg_check:6.1f}%  | {agg_bet:6.1f}%")

# IP after OOP check
print("\n--- IP after OOP Check ---")
total_check = 0
total_bet = 0
total_deals = 0
hand_groups = {}

for hand in solver.ip_hands:
    deals = ip_hand_deals.get(hand, 0)

    c1_rank = hand[0] // 4
    c2_rank = hand[1] // 4

    if c1_rank == c2_rank:
        hand_type = f"{['2','3','4','5','6','7','8','9','T','J','Q','K','A'][c1_rank]}{['2','3','4','5','6','7','8','9','T','J','Q','K','A'][c1_rank]}"
    else:
        r1 = ['2','3','4','5','6','7','8','9','T','J','Q','K','A'][max(c1_rank, c2_rank)]
        r2 = ['2','3','4','5','6','7','8','9','T','J','Q','K','A'][min(c1_rank, c2_rank)]
        hand_type = f"{r1}{r2}"

    if hand_type not in hand_groups:
        hand_groups[hand_type] = {'check': 0, 'bet': 0, 'deals': 0}

    strat = solver.get_strategy_for_hand(1, hand, action_id=1)
    check = strat.get('Check', 0)
    bet = strat.get('All-in', 0)

    hand_groups[hand_type]['check'] += check * deals
    hand_groups[hand_type]['bet'] += bet * deals
    hand_groups[hand_type]['deals'] += deals

    total_check += check * deals
    total_bet += bet * deals
    total_deals += deals

print(f"{'Hand':<8} | {'Deals':<6} | {'Check':<8} | {'All-in':<8}")
print("-" * 45)
for ht in sorted(hand_groups.keys(), key=lambda x: -hand_groups[x]['deals']):
    g = hand_groups[ht]
    avg_check = g['check'] / g['deals'] * 100
    avg_bet = g['bet'] / g['deals'] * 100
    print(f"{ht:<8} | {g['deals']:<6} | {avg_check:6.1f}%  | {avg_bet:6.1f}%")

print("-" * 45)
agg_check = total_check / total_deals * 100
agg_bet = total_bet / total_deals * 100
print(f"{'TOTAL':<8} | {total_deals:<6} | {agg_check:6.1f}%  | {agg_bet:6.1f}%")

print("\n" + "=" * 60)
print("EXPECTED (from wasm-postflop):")
print("  OOP: 100% Check")
print("  IP after check: 35.5% Check, 64.5% All-in")
print("=" * 60)
