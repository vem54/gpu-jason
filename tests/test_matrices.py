"""
Tests for matrix building from game trees.

Run with: pytest tests/test_matrices.py -v
"""

import pytest
import numpy as np
from scipy import sparse

from gpu_poker_cfr.games.kuhn import KuhnPoker
from gpu_poker_cfr.matrix.builder import (
    build_adjacency_matrix,
    build_level_graphs,
    build_game_matrices,
    print_matrix_stats
)


class TestAdjacencyMatrix:
    """Test adjacency matrix G construction."""

    @pytest.fixture
    def tree(self):
        return KuhnPoker().build_tree()

    def test_shape(self, tree):
        """G should be square with shape (num_nodes, num_nodes)."""
        G = build_adjacency_matrix(tree)
        assert G.shape == (tree.num_nodes, tree.num_nodes)

    def test_is_sparse_csr(self, tree):
        """G should be in CSR format."""
        G = build_adjacency_matrix(tree)
        assert sparse.isspmatrix_csr(G)

    def test_num_edges(self, tree):
        """Number of edges = num_nodes - 1 (tree has n-1 edges)."""
        G = build_adjacency_matrix(tree)
        assert G.nnz == tree.num_nodes - 1

    def test_root_has_no_incoming_edges(self, tree):
        """Root node should have no incoming edges (column 0 is zero)."""
        G = build_adjacency_matrix(tree)
        # Root is node 0, check its column
        root_col = G.getcol(0).toarray().flatten()
        assert np.sum(root_col) == 0

    def test_parent_child_relationship(self, tree):
        """G[parent, child] = 1 for all parent-child pairs."""
        G = build_adjacency_matrix(tree)
        for node_id, parent_id in enumerate(tree.parent_ids):
            if parent_id >= 0:
                assert G[parent_id, node_id] == 1.0


class TestLevelGraphs:
    """Test level graph L construction."""

    @pytest.fixture
    def tree(self):
        return KuhnPoker().build_tree()

    def test_num_level_graphs(self, tree):
        """Should have max_depth level graphs."""
        L = build_level_graphs(tree)
        assert len(L) == tree.max_depth

    def test_all_sparse_csr(self, tree):
        """All level graphs should be CSR sparse."""
        L = build_level_graphs(tree)
        for L_l in L:
            assert sparse.isspmatrix_csr(L_l)

    def test_edges_sum_to_total(self, tree):
        """Sum of edges across all levels = total edges."""
        G = build_adjacency_matrix(tree)
        L = build_level_graphs(tree)
        total_level_edges = sum(L_l.nnz for L_l in L)
        assert total_level_edges == G.nnz

    def test_level_edges_connect_correct_depths(self, tree):
        """L[l] should only connect depth l-1 to depth l."""
        L = build_level_graphs(tree)

        for level_idx, L_l in enumerate(L):
            level = level_idx + 1  # L[0] is level 1

            # Get the edges
            rows, cols = L_l.nonzero()

            for parent_id, child_id in zip(rows, cols):
                parent_depth = tree.depths[parent_id]
                child_depth = tree.depths[child_id]

                assert parent_depth == level - 1, f"Parent at wrong depth for level {level}"
                assert child_depth == level, f"Child at wrong depth for level {level}"


class TestInfosetActionMapping:
    """Test M_QV and M_HQ matrix construction."""

    @pytest.fixture
    def matrices(self):
        tree = KuhnPoker().build_tree()
        return build_game_matrices(tree)

    def test_m_qv_shape(self, matrices):
        """M_QV should be (num_infoset_actions, num_nodes)."""
        assert matrices.M_QV.shape == (matrices.num_infoset_actions, matrices.num_nodes)

    def test_m_hq_shape(self, matrices):
        """M_HQ should be (num_infosets, num_infoset_actions)."""
        assert matrices.M_HQ.shape == (matrices.num_infosets, matrices.num_infoset_actions)

    def test_num_infoset_actions_kuhn(self, matrices):
        """Kuhn has 12 infosets × 2 actions each = 24 infoset-action pairs."""
        assert matrices.num_infoset_actions == 24

    def test_m_hq_row_sums(self, matrices):
        """Each infoset row in M_HQ should sum to number of actions."""
        row_sums = np.array(matrices.M_HQ.sum(axis=1)).flatten()
        # All Kuhn infosets have 2 actions
        assert np.all(row_sums == 2)

    def test_infoset_action_to_infoset_valid(self, matrices):
        """Infoset-action mapping should be valid."""
        assert len(matrices.infoset_action_to_infoset) == matrices.num_infoset_actions
        assert np.all(matrices.infoset_action_to_infoset >= 0)
        assert np.all(matrices.infoset_action_to_infoset < matrices.num_infosets)


class TestPlayerMapping:
    """Test M_VI matrix construction."""

    @pytest.fixture
    def matrices(self):
        tree = KuhnPoker().build_tree()
        return build_game_matrices(tree)

    def test_m_vi_shape(self, matrices):
        """M_VI should be (num_nodes, num_players)."""
        assert matrices.M_VI.shape == (matrices.num_nodes, matrices.num_players)

    def test_m_vi_is_dense(self, matrices):
        """M_VI should be a dense numpy array."""
        assert isinstance(matrices.M_VI, np.ndarray)

    def test_m_vi_binary(self, matrices):
        """M_VI values should be 0 or 1."""
        assert np.all((matrices.M_VI == 0) | (matrices.M_VI == 1))

    def test_root_has_no_parent_player(self, matrices):
        """Root node should have no acting player."""
        assert np.sum(matrices.M_VI[0]) == 0


class TestFullGameMatrices:
    """Test complete GameMatrices construction."""

    @pytest.fixture
    def matrices(self):
        tree = KuhnPoker().build_tree()
        return build_game_matrices(tree)

    def test_terminal_mappings_consistent(self, matrices):
        """Terminal index mappings should be consistent."""
        for term_idx, node_idx in enumerate(matrices.terminal_node_indices):
            assert matrices.node_to_terminal_idx[node_idx] == term_idx

    def test_terminal_utilities_shape(self, matrices):
        """Terminal utilities should have correct shape."""
        assert matrices.terminal_utilities.shape == (matrices.num_terminals, matrices.num_players)

    def test_chance_probs_shape(self, matrices):
        """Chance probs should have correct shape."""
        assert matrices.chance_probs.shape == (matrices.num_nodes,)

    def test_infoset_action_offsets_valid(self, matrices):
        """Infoset action offsets should be valid."""
        assert len(matrices.infoset_action_offsets) == matrices.num_infosets + 1
        assert matrices.infoset_action_offsets[0] == 0
        assert matrices.infoset_action_offsets[-1] == matrices.num_infoset_actions


class TestMatrixSparsity:
    """Test that matrices have expected sparsity patterns."""

    @pytest.fixture
    def matrices(self):
        tree = KuhnPoker().build_tree()
        return build_game_matrices(tree)

    def test_g_is_sparse(self, matrices):
        """G should be highly sparse for any reasonable game."""
        g_sparsity = 100 * (1 - matrices.G.nnz / (matrices.num_nodes ** 2))
        assert g_sparsity > 90, f"G sparsity {g_sparsity}% is too low"

    def test_m_qv_is_sparse(self, matrices):
        """M_QV should be highly sparse."""
        total_elements = matrices.M_QV.shape[0] * matrices.M_QV.shape[1]
        sparsity = 100 * (1 - matrices.M_QV.nnz / total_elements)
        assert sparsity > 90, f"M_QV sparsity {sparsity}% is too low"


def test_matrix_stats_output(capsys):
    """Test that print_matrix_stats runs without error."""
    tree = KuhnPoker().build_tree()
    matrices = build_game_matrices(tree)
    print_matrix_stats(matrices)

    captured = capsys.readouterr()
    assert "Game Matrix Statistics" in captured.out
    assert "Nodes: 61" in captured.out


def test_kuhn_matrices_smoke():
    """Quick smoke test for full matrix building."""
    tree = KuhnPoker().build_tree()
    matrices = build_game_matrices(tree)

    assert matrices.num_nodes == 61
    assert matrices.num_terminals == 30
    assert matrices.num_infosets == 12
    assert matrices.num_infoset_actions == 24
    assert matrices.max_depth == 5

    print("\nKuhn Matrix Stats:")
    print(f"  G: {matrices.G.shape}, nnz={matrices.G.nnz}")
    print(f"  L: {len(matrices.L)} levels")
    print(f"  M_QV: {matrices.M_QV.shape}, nnz={matrices.M_QV.nnz}")
    print(f"  M_HQ: {matrices.M_HQ.shape}, nnz={matrices.M_HQ.nnz}")
    print(f"  M_VI: {matrices.M_VI.shape}")
