"""Debug AA hand specifically."""

import cupy as cp
import numpy as np
from gpu_poker_cfr.games.custom_river_toy import make_aa_kk_vs_ak_aq_game
from gpu_poker_cfr.games.cards import card_name
from gpu_poker_cfr.solvers.gpu_custom_river import GPUCustomRiverCFR
from gpu_poker_cfr.games.hand_eval import evaluate_7cards, HAND_NAMES

game = make_aa_kk_vs_ak_aq_game()
solver = GPUCustomRiverCFR(game)

board = list(game.board)

# Find IP's AhAd
hand = None
for h in solver.ip_hands:
    if card_name(h[0]) == 'Ah' and card_name(h[1]) == 'Ad':
        hand = h
        break
    if card_name(h[1]) == 'Ah' and card_name(h[0]) == 'Ad':
        hand = h
        break

if hand is None:
    print("Could not find AhAd in IP hands")
    print("IP hands:", [(card_name(h[0]) + card_name(h[1])) for h in solver.ip_hands[:10]])
    exit()

hand_str = f"{card_name(hand[0])}{card_name(hand[1])}"
print(f"Checking IP hand: {hand_str}")

# Find matching deals and analyze matchups
matching_deals = []
for d in range(solver.num_deals):
    oop_hand, ip_hand = game.get_deal(d)
    ip_hand_norm = tuple(sorted(ip_hand, reverse=True))
    if ip_hand_norm == hand:
        matching_deals.append(d)

print(f"Matching deals: {len(matching_deals)}")

# Analyze what OOP hands we face and win/lose against
print("\nMatchup analysis for IP AhAd:")
wins = 0
losses = 0
ties = 0

for d in matching_deals:
    oop_hand, ip_hand = game.get_deal(d)
    oop_str = f"{card_name(oop_hand[0])}{card_name(oop_hand[1])}"

    # Evaluate both hands
    ip_cards = list(ip_hand) + board
    oop_cards = list(oop_hand) + board
    ip_val = evaluate_7cards(np.array(ip_cards, dtype=np.int32))
    oop_val = evaluate_7cards(np.array(oop_cards, dtype=np.int32))

    ip_type = HAND_NAMES[ip_val >> 20]
    oop_type = HAND_NAMES[oop_val >> 20]

    if ip_val > oop_val:
        result = "WIN"
        wins += 1
    elif ip_val < oop_val:
        result = "LOSE"
        losses += 1
    else:
        result = "TIE"
        ties += 1

print(f"Wins: {wins}, Losses: {losses}, Ties: {ties}")
print(f"Win rate: {wins/(wins+losses+ties)*100:.1f}%")

# What should the strategy be?
# If IP calls 100 to win 200 pot when ahead:
# EV(call) = P(win)*100 - P(lose)*100
ev_call = (wins * 100 - losses * 100) / (wins + losses + ties)
print(f"EV(call) = {ev_call:.1f}")
print(f"Since EV(call) = {ev_call:.1f} > 0, IP should always call")

# Now check what the solver computed
solver.solve(1000)

print("\n--- Solver results ---")
strat1 = solver.get_strategy_for_hand(1, hand, action_id=1)
strat6 = solver.get_strategy_for_hand(1, hand, action_id=6)

print(f"After OOP check: {strat1}")
print(f"After OOP all-in: {strat6}")
