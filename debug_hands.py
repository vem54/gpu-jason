"""Debug hand evaluation on the board."""

import numpy as np
from gpu_poker_cfr.games.cards import make_card, card_name, RANK_A, RANK_K, RANK_Q, RANK_J, RANK_T, RANK_5, CLUBS, SPADES, HEARTS, DIAMONDS
from gpu_poker_cfr.games.hand_eval import evaluate_7cards, HAND_NAMES

# Board: As Ks Qs Th 5c
board = [
    make_card(RANK_A, SPADES),   # As
    make_card(RANK_K, SPADES),   # Ks
    make_card(RANK_Q, SPADES),   # Qs
    make_card(RANK_T, HEARTS),   # Th
    make_card(RANK_5, CLUBS),    # 5c
]

print("Board:", " ".join(card_name(c) for c in board))
print()

# Test hands
test_hands = [
    ("AA", [make_card(RANK_A, HEARTS), make_card(RANK_A, DIAMONDS)]),
    ("KK", [make_card(RANK_K, HEARTS), make_card(RANK_K, DIAMONDS)]),
    ("JJ", [make_card(RANK_J, HEARTS), make_card(RANK_J, DIAMONDS)]),
    ("55", [make_card(RANK_5, HEARTS), make_card(RANK_5, DIAMONDS)]),
    ("AK", [make_card(RANK_A, HEARTS), make_card(RANK_K, HEARTS)]),
    ("AQ", [make_card(RANK_A, HEARTS), make_card(RANK_Q, HEARTS)]),
    ("AJ", [make_card(RANK_A, HEARTS), make_card(RANK_J, HEARTS)]),
    ("AT", [make_card(RANK_A, HEARTS), make_card(RANK_T, DIAMONDS)]),
    ("JT", [make_card(RANK_J, HEARTS), make_card(RANK_T, DIAMONDS)]),
]

print("Hand evaluations:")
print("-" * 50)

results = []
for name, hand in test_hands:
    cards_7 = hand + board
    value = evaluate_7cards(np.array(cards_7, dtype=np.int32))
    hand_type = value >> 20
    results.append((name, value, hand_type))
    print(f"{name}: {card_name(hand[0])}{card_name(hand[1])} -> value={value}, type={HAND_NAMES[hand_type]}")

print()
print("Ranking (higher value = better hand):")
print("-" * 50)
for name, value, hand_type in sorted(results, key=lambda x: x[1], reverse=True):
    print(f"{name}: {value} ({HAND_NAMES[hand_type]})")
