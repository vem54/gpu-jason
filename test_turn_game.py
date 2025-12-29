"""Test turn toy game structure."""

from gpu_poker_cfr.games.turn_toy_game import (
    make_turn_toy_game,
    build_turn_tree,
    print_tree_structure
)

print("Creating turn toy game...")
game = make_turn_toy_game()

print(f"\nGame board: {game.board}")
print(f"Num deals: {game.num_deals}")
print(f"Possible rivers: {game.num_rivers}")

print("\nBuilding tree...")
nodes, child_indices = build_turn_tree(game)

print_tree_structure(nodes, child_indices)

# Count node types
from collections import Counter
type_counts = Counter(n['type'] for n in nodes)
print(f"\nNode type counts: {dict(type_counts)}")
print(f"  FOLD: {type_counts[0]}")
print(f"  SHOWDOWN: {type_counts[1]}")
print(f"  DECISION: {type_counts[2]}")
print(f"  CHANCE: {type_counts[3]}")
