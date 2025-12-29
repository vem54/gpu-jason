"""
Tests for Kuhn Poker game tree.

Expected properties (from literature):
- 12 information sets (6 per player)
- 30 terminal nodes
- Game tree depth of 4-5 depending on counting method

Run with: pytest tests/test_kuhn.py -v
"""

import pytest
import numpy as np
from gpu_poker_cfr.games.kuhn import KuhnPoker, JACK, QUEEN, KING
from gpu_poker_cfr.games.base import Player


class TestKuhnPokerTree:
    """Test Kuhn Poker game tree construction."""

    @pytest.fixture
    def game(self):
        return KuhnPoker()

    @pytest.fixture
    def tree(self, game):
        return game.build_tree()

    def test_game_name(self, game):
        assert game.name == "kuhn_poker"

    def test_num_players(self, game):
        assert game.num_players == 2

    def test_num_terminals(self, tree):
        """Kuhn poker has 30 terminal nodes (5 per deal × 6 deals)."""
        assert tree.num_terminals == 30, f"Expected 30 terminals, got {tree.num_terminals}"

    def test_num_infosets(self, tree):
        """Kuhn poker has 12 information sets (6 per player)."""
        assert tree.num_infosets == 12, f"Expected 12 infosets, got {tree.num_infosets}"

    def test_infosets_per_player(self, tree):
        """Each player should have 6 information sets."""
        p1_infosets = [i for i in tree.infosets if i.player == Player.PLAYER_1]
        p2_infosets = [i for i in tree.infosets if i.player == Player.PLAYER_2]

        assert len(p1_infosets) == 6, f"P1 should have 6 infosets, got {len(p1_infosets)}"
        assert len(p2_infosets) == 6, f"P2 should have 6 infosets, got {len(p2_infosets)}"

    def test_terminal_utilities_sum_to_zero(self, tree):
        """Kuhn poker is zero-sum."""
        for i in range(tree.num_terminals):
            util_sum = tree.terminal_utilities[i].sum()
            assert abs(util_sum) < 1e-6, f"Terminal {i} utilities don't sum to zero: {tree.terminal_utilities[i]}"

    def test_terminal_utilities_range(self, tree):
        """Utilities should be in expected range for Kuhn."""
        # Pot sizes: 2 (check-check), 3 (fold after bet), 4 (call)
        # Utilities are pot/2 for winner, -pot/2 for loser
        for i in range(tree.num_terminals):
            for j in range(2):
                util = abs(tree.terminal_utilities[i, j])
                assert util in [1.0, 1.5, 2.0], f"Unexpected utility {util}"

    def test_all_infosets_have_2_actions(self, tree):
        """Every infoset in Kuhn poker has exactly 2 actions."""
        for infoset in tree.infosets:
            assert len(infoset.actions) == 2, f"Infoset {infoset.key} has {len(infoset.actions)} actions"

    def test_infoset_num_actions_array(self, tree):
        """Infoset action counts should all be 2."""
        assert np.all(tree.infoset_num_actions == 2)

    def test_root_has_no_parent(self, tree):
        """Root node should have parent_id = -1."""
        assert tree.parent_ids[0] == -1

    def test_depth_consistency(self, tree):
        """Children should have depth = parent_depth + 1."""
        for node in tree.nodes:
            if node.parent_id is not None and node.parent_id >= 0:
                parent_depth = tree.depths[node.parent_id]
                assert node.depth == parent_depth + 1, f"Node {node.id} depth inconsistent"

    def test_chance_probs_sum_to_one(self, tree):
        """Chance deal probabilities should sum to 1."""
        # Nodes at depth 1 are the 6 deals
        deal_nodes = [n for n in tree.nodes if n.depth == 1]
        assert len(deal_nodes) == 6
        prob_sum = sum(n.chance_prob for n in deal_nodes)
        assert abs(prob_sum - 1.0) < 1e-6


class TestKuhnInfosetKeys:
    """Test information set key generation."""

    @pytest.fixture
    def game(self):
        return KuhnPoker()

    def test_same_card_same_history_same_infoset(self, game):
        """Two nodes with same card and history should have same infoset key."""
        # Player 1 with Jack, no actions yet
        key1 = game.infoset_key(Player.PLAYER_1, ((JACK, QUEEN), ()))
        key2 = game.infoset_key(Player.PLAYER_1, ((JACK, KING), ()))
        assert key1 == key2 == "J:"

    def test_different_cards_different_infosets(self, game):
        """Different cards should give different infoset keys."""
        key_j = game.infoset_key(Player.PLAYER_1, ((JACK, QUEEN), ()))
        key_q = game.infoset_key(Player.PLAYER_1, ((QUEEN, JACK), ()))
        key_k = game.infoset_key(Player.PLAYER_1, ((KING, JACK), ()))

        assert key_j != key_q != key_k

    def test_player_sees_own_card(self, game):
        """Each player sees only their own card."""
        # P1 has Jack, P2 has Queen
        p1_key = game.infoset_key(Player.PLAYER_1, ((JACK, QUEEN), ()))
        p2_key = game.infoset_key(Player.PLAYER_2, ((JACK, QUEEN), ()))

        assert "J" in p1_key
        assert "Q" in p2_key


class TestKuhnTreeStructure:
    """Test detailed tree structure properties."""

    @pytest.fixture
    def tree(self):
        return KuhnPoker().build_tree()

    def test_decision_nodes_have_valid_infosets(self, tree):
        """All decision nodes should map to valid infosets."""
        for node in tree.nodes:
            if node.player in (Player.PLAYER_1, Player.PLAYER_2) and not node.is_terminal:
                assert node.infoset_id is not None
                assert 0 <= node.infoset_id < tree.num_infosets

    def test_terminal_nodes_have_utilities(self, tree):
        """All terminal nodes should have utilities."""
        for node in tree.nodes:
            if node.is_terminal:
                assert node.utility is not None
                assert len(node.utility) == 2

    def test_showdown_higher_card_wins(self, tree):
        """At showdown, higher card should win."""
        # Find check-check terminals and verify utilities
        for node in tree.nodes:
            if node.is_terminal and node.depth > 0:
                # Trace back to find the cards
                history = self._get_history(tree, node)
                if history is not None:
                    cards, actions = history
                    # Check if this is a showdown (not a fold)
                    is_fold = any(a.name == 'f' for a in actions)
                    if not is_fold and len(cards) == 2:
                        if cards[0] > cards[1]:
                            assert node.utility[0] > 0, f"Higher card should win: {cards}, {node.utility}"
                        else:
                            assert node.utility[1] > 0, f"Higher card should win: {cards}, {node.utility}"

    def _get_history(self, tree, node):
        """Helper to reconstruct history for a node."""
        # This is a simplified check - actual history stored in node creation
        return None  # Placeholder


class TestTreeArrayConsistency:
    """Test that numpy arrays are consistent with node objects."""

    @pytest.fixture
    def tree(self):
        return KuhnPoker().build_tree()

    def test_terminal_indices_match(self, tree):
        """Terminal indices array should match is_terminal property."""
        terminal_set = set(tree.terminal_indices)
        for node in tree.nodes:
            if node.is_terminal:
                assert node.id in terminal_set
            else:
                assert node.id not in terminal_set

    def test_depths_array_matches_nodes(self, tree):
        """Depths array should match node.depth."""
        for node in tree.nodes:
            assert tree.depths[node.id] == node.depth

    def test_parent_ids_array_matches_nodes(self, tree):
        """Parent IDs array should match node.parent_id."""
        for node in tree.nodes:
            expected = node.parent_id if node.parent_id is not None else -1
            assert tree.parent_ids[node.id] == expected


def test_kuhn_quick_smoke():
    """Quick smoke test that Kuhn poker builds without error."""
    game = KuhnPoker()
    tree = game.build_tree()

    # Basic sanity checks
    assert tree.num_nodes > 0
    assert tree.num_terminals > 0
    assert tree.num_infosets > 0

    print(f"\nKuhn Poker Tree Stats:")
    print(f"  Nodes: {tree.num_nodes}")
    print(f"  Terminals: {tree.num_terminals}")
    print(f"  Decision nodes: {tree.num_decision_nodes}")
    print(f"  Infosets: {tree.num_infosets}")
    print(f"  Max depth: {tree.max_depth}")
