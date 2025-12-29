"""Debug script to trace CFR calculation for a specific deal."""

import numpy as np
from gpu_poker_cfr.games.turn_toy_game import make_turn_toy_game, build_turn_tree
from gpu_poker_cfr.games.cards import card_name, make_card, RANK_J, RANK_A, CLUBS, SPADES, DIAMONDS
from gpu_poker_cfr.games.hand_eval import evaluate_7cards

# Create game
game = make_turn_toy_game()
nodes, children = build_turn_tree(game)
num_nodes = len(nodes)

# Find deal: OOP=AcAd, IP=JsJc
oop_target = tuple(sorted([make_card(RANK_A, CLUBS), make_card(RANK_A, DIAMONDS)], reverse=True))
ip_target = tuple(sorted([make_card(RANK_J, SPADES), make_card(RANK_J, CLUBS)], reverse=True))

deal_idx = None
for d in range(game.num_deals):
    oop_hand, ip_hand = game.get_deal(d)
    if tuple(sorted(oop_hand, reverse=True)) == oop_target and tuple(sorted(ip_hand, reverse=True)) == ip_target:
        deal_idx = d
        break

print(f"Deal {deal_idx}: OOP={card_name(oop_target[0])}{card_name(oop_target[1])}, IP={card_name(ip_target[0])}{card_name(ip_target[1])}")

# Precompute river hand values for this deal
board = list(game.board)
valid_rivers = game.get_valid_rivers(deal_idx)
river_p0_values = {}
river_p1_values = {}
for r in valid_rivers:
    h0 = np.array(list(oop_target) + board + [r], dtype=np.int32)
    h1 = np.array(list(ip_target) + board + [r], dtype=np.int32)
    river_p0_values[r] = evaluate_7cards(h0)
    river_p1_values[r] = evaluate_7cards(h1)

# Manual CFR iteration
# Use uniform strategy for simplicity
strategy = {}
for i, node in enumerate(nodes):
    if node['type'] == 2:  # DECISION
        n_act = len(node['actions'])
        strategy[i] = [1.0 / n_act] * n_act

print(f"\n=== Manual CFR Iteration for Deal {deal_idx} ===")
print(f"Valid rivers: {len(valid_rivers)}")

# Forward pass: compute reaches
reach0 = np.zeros(num_nodes)
reach1 = np.zeros(num_nodes)
reach0[0] = 1.0
reach1[0] = 1.0

for i in range(num_nodes):
    node = nodes[i]
    if node['type'] == 3:  # CHANCE
        child = children[i, 0]
        if child >= 0:
            reach0[child] = reach0[i]
            reach1[child] = reach1[i]
    elif node['type'] == 2:  # DECISION
        player = node['player']
        for a, child in enumerate(children[i]):
            if child >= 0:
                s = strategy[i][a]
                if player == 0:
                    reach0[child] = reach0[i] * s
                    reach1[child] = reach1[i]
                else:
                    reach0[child] = reach0[i]
                    reach1[child] = reach1[i] * s

print(f"\nReaches at key nodes:")
print(f"  Node 0 (OOP turn root): reach0={reach0[0]:.3f}, reach1={reach1[0]:.3f}")
print(f"  Node 1 (IP after check): reach0={reach0[1]:.3f}, reach1={reach1[1]:.3f}")
print(f"  Node 4 (OOP facing all-in): reach0={reach0[4]:.3f}, reach1={reach1[4]:.3f}")

# Proper backward pass with all rivers
print(f"\n=== Full Backward Pass (averaging over {len(valid_rivers)} rivers) ===")

# We need to properly average chance node EVs over all rivers
# River nodes: 7, 10-23 (is_river flag)
river_nodes = [7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
turn_nodes = [0, 1, 2, 3, 4, 5, 6, 8, 9]

# Accumulate chance node EVs
chance_ev0 = {3: 0.0, 6: 0.0, 9: 0.0}  # Chance nodes
chance_ev1 = {3: 0.0, 6: 0.0, 9: 0.0}

river_weight = 1.0 / len(valid_rivers)

for river in valid_rivers:
    p0_val = river_p0_values[river]
    p1_val = river_p1_values[river]

    ev0 = np.zeros(num_nodes)
    ev1 = np.zeros(num_nodes)

    # Backward pass for river nodes only
    for i in range(num_nodes - 1, -1, -1):
        if i not in river_nodes:
            continue
        node = nodes[i]
        t = node['type']

        if t == 0:  # FOLD
            fp = node['fold_player']
            inv_oop = node['invested_oop']
            inv_ip = node['invested_ip']
            if fp == 0:
                ev0[i] = -inv_oop
                ev1[i] = inv_oop
            else:
                ev0[i] = inv_ip
                ev1[i] = -inv_ip

        elif t == 1:  # SHOWDOWN
            inv_oop = node['invested_oop']
            inv_ip = node['invested_ip']
            if p0_val > p1_val:
                ev0[i] = inv_ip
                ev1[i] = -inv_ip
            elif p1_val > p0_val:
                ev0[i] = -inv_oop
                ev1[i] = inv_oop
            else:
                ev0[i] = 0
                ev1[i] = 0

        elif t == 2:  # DECISION
            player = node['player']
            n_act = len(node['actions'])
            for a in range(n_act):
                child = children[i, a]
                if child >= 0:
                    s = strategy[i][a]
                    ev0[i] += s * ev0[child]
                    ev1[i] += s * ev1[child]

    # Accumulate chance node EVs
    for c in [3, 6, 9]:
        child = children[c, 0]
        if child >= 0:
            chance_ev0[c] += ev0[child] * river_weight
            chance_ev1[c] += ev1[child] * river_weight

# Now backward pass for turn nodes using averaged chance EVs
ev0 = np.zeros(num_nodes)
ev1 = np.zeros(num_nodes)

for i in range(num_nodes - 1, -1, -1):
    if i in river_nodes:
        continue
    node = nodes[i]
    t = node['type']

    if t == 0:  # FOLD
        fp = node['fold_player']
        inv_oop = node['invested_oop']
        inv_ip = node['invested_ip']
        if fp == 0:
            ev0[i] = -inv_oop
            ev1[i] = inv_oop
        else:
            ev0[i] = inv_ip
            ev1[i] = -inv_ip

    elif t == 3:  # CHANCE
        ev0[i] = chance_ev0[i]
        ev1[i] = chance_ev1[i]

    elif t == 2:  # DECISION
        player = node['player']
        n_act = len(node['actions'])
        for a in range(n_act):
            child = children[i, a]
            if child >= 0:
                s = strategy[i][a]
                ev0[i] += s * ev0[child]
                ev1[i] += s * ev1[child]

print(f"Chance node EVs (averaged over rivers):")
for c in [3, 6, 9]:
    print(f"  Node {c}: ev0={chance_ev0[c]:.1f}, ev1={chance_ev1[c]:.1f}")

print(f"\nKey node EVs:")
for i in [0, 1, 2, 4]:
    print(f"  Node {i}: ev0={ev0[i]:.1f}, ev1={ev1[i]:.1f}")

# Compute regret at node 1 (IP decision)
print(f"\n=== Regret at Node 1 (IP decision) ===")
node = nodes[1]
player = 1  # IP
n_act = 2  # Check, All-in
opp_reach = reach0[1]

node_ev1 = ev1[1]
for a in range(n_act):
    child = children[1, a]
    child_ev1 = ev1[child] if child < len(ev0) else chance_ev1.get(child, 0)
    regret = (child_ev1 - node_ev1) * opp_reach
    print(f"  {nodes[1]['actions'][a]}: child={child}, child_ev1={child_ev1:.1f}, regret={regret:.3f}")
