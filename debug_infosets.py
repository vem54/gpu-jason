"""Debug infoset mapping."""

from gpu_poker_cfr.games.custom_river_toy import make_aa_kk_vs_ak_aq_game
from gpu_poker_cfr.games.cards import card_name
from gpu_poker_cfr.solvers.gpu_custom_river import GPUCustomRiverCFR

game = make_aa_kk_vs_ak_aq_game()
solver = GPUCustomRiverCFR(game)

print("Tree nodes:")
for i, node in enumerate(solver.nodes):
    print(f"  Node {i}: type={node['type']}, action_id={node.get('action_id')}, player={node.get('player')}")

print()
print(f"Number of infosets: {solver.num_infosets}")
print(f"OOP hands: {len(solver.oop_hands)}")
print(f"IP hands: {len(solver.ip_hands)}")

print()
print("Infoset key to index (first 30):")
for i, (key, idx) in enumerate(sorted(solver.infoset_key_to_idx.items(), key=lambda x: x[1])):
    if i >= 30:
        break
    player, hand_idx, action_id = key
    if player == 0:
        hand = solver.oop_hands[hand_idx]
    else:
        hand = solver.ip_hands[hand_idx]
    hand_str = f"{card_name(hand[0])}{card_name(hand[1])}"
    player_name = "OOP" if player == 0 else "IP"
    print(f"  Infoset {idx}: {player_name} {hand_str} at action_id={action_id}")

# Check if action_id 1 and 6 share any infosets
print()
print("Checking IP infosets at action_id 1 vs 6:")
for hand_idx in range(min(3, len(solver.ip_hands))):
    hand = solver.ip_hands[hand_idx]
    hand_str = f"{card_name(hand[0])}{card_name(hand[1])}"

    key1 = (1, hand_idx, 1)  # IP at node 1
    key6 = (1, hand_idx, 6)  # IP at node 6

    idx1 = solver.infoset_key_to_idx.get(key1, -1)
    idx6 = solver.infoset_key_to_idx.get(key6, -1)

    print(f"  IP {hand_str}: action_id=1 -> infoset {idx1}, action_id=6 -> infoset {idx6}")
