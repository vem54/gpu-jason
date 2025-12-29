"""
Tests for CFR operations (forward/backward passes, regret matching).

Run with: pytest tests/test_ops.py -v
"""

import pytest
import numpy as np

from gpu_poker_cfr.games.kuhn import KuhnPoker
from gpu_poker_cfr.matrix.builder import build_game_matrices
from gpu_poker_cfr.engine.backend import get_backend
from gpu_poker_cfr.engine.ops import (
    forward_reach_simple,
    backward_values_simple,
    compute_counterfactual_values,
    compute_infoset_cf_values,
    compute_instant_regret,
    regret_match,
    uniform_strategy,
)


class TestUniformStrategy:
    """Test uniform strategy creation."""

    @pytest.fixture
    def setup(self):
        tree = KuhnPoker().build_tree()
        matrices = build_game_matrices(tree)
        backend = get_backend('numpy')
        return matrices, backend

    def test_shape(self, setup):
        matrices, backend = setup
        strategy = uniform_strategy(matrices, backend)
        assert strategy.shape == (matrices.num_infoset_actions,)

    def test_is_valid_distribution(self, setup):
        matrices, backend = setup
        strategy = uniform_strategy(matrices, backend)
        strategy = backend.asnumpy(strategy)

        # Check each infoset sums to 1
        for h_idx in range(matrices.num_infosets):
            start = matrices.infoset_action_offsets[h_idx]
            end = matrices.infoset_action_offsets[h_idx + 1]
            assert np.isclose(strategy[start:end].sum(), 1.0)

    def test_uniform_values(self, setup):
        matrices, backend = setup
        strategy = uniform_strategy(matrices, backend)
        strategy = backend.asnumpy(strategy)

        # All Kuhn infosets have 2 actions, so each should be 0.5
        for h_idx in range(matrices.num_infosets):
            start = matrices.infoset_action_offsets[h_idx]
            end = matrices.infoset_action_offsets[h_idx + 1]
            expected = 1.0 / (end - start)
            assert np.allclose(strategy[start:end], expected)


class TestForwardReach:
    """Test forward reach probability computation."""

    @pytest.fixture
    def setup(self):
        tree = KuhnPoker().build_tree()
        matrices = build_game_matrices(tree)
        backend = get_backend('numpy')
        strategy = uniform_strategy(matrices, backend)
        return matrices, backend, strategy

    def test_shape(self, setup):
        matrices, backend, strategy = setup
        reach = forward_reach_simple(strategy, matrices, backend)
        assert reach.shape == (matrices.num_nodes, matrices.num_players)

    def test_root_reach_is_one(self, setup):
        matrices, backend, strategy = setup
        reach = forward_reach_simple(strategy, matrices, backend)
        reach = backend.asnumpy(reach)

        # Root has reach 1.0 for both players
        assert reach[0, 0] == 1.0
        assert reach[0, 1] == 1.0

    def test_reach_decreases_or_stays(self, setup):
        matrices, backend, strategy = setup
        reach = forward_reach_simple(strategy, matrices, backend)
        reach = backend.asnumpy(reach)

        # Reach probabilities should be <= 1 and >= 0
        assert np.all(reach >= 0)
        assert np.all(reach <= 1.0)

    def test_reach_decreases_with_actions(self, setup):
        """When a player takes an action, their reach decreases (uniform = 0.5 per action)."""
        matrices, backend, strategy = setup
        reach = forward_reach_simple(strategy, matrices, backend)
        reach = backend.asnumpy(reach)

        # In Kuhn poker:
        # Depth 0: Root
        # Depth 1: Chance deals P1's card
        # Depth 2: Chance deals P2's card
        # Depth 3+: Players act
        # So depth >= 3 means at least one player has acted
        for node_id in range(1, matrices.num_nodes):
            depth = matrices.depths[node_id]
            if depth >= 3:  # After at least one player action
                # At least one player should have reach < 1
                assert reach[node_id, 0] < 1.0 or reach[node_id, 1] < 1.0, \
                    f"Node {node_id} at depth {depth} has reach {reach[node_id]}"


class TestBackwardValues:
    """Test backward value propagation."""

    @pytest.fixture
    def setup(self):
        tree = KuhnPoker().build_tree()
        matrices = build_game_matrices(tree)
        backend = get_backend('numpy')
        strategy = uniform_strategy(matrices, backend)
        reach = forward_reach_simple(strategy, matrices, backend)
        return matrices, backend, strategy, reach

    def test_shape(self, setup):
        matrices, backend, strategy, reach = setup
        values = backward_values_simple(strategy, reach, matrices, backend)
        assert values.shape == (matrices.num_nodes, matrices.num_players)

    def test_terminal_values_match(self, setup):
        """Terminal values should equal terminal utilities."""
        matrices, backend, strategy, reach = setup
        values = backward_values_simple(strategy, reach, matrices, backend)
        values = backend.asnumpy(values)

        for term_idx, node_idx in enumerate(matrices.terminal_node_indices):
            expected = matrices.terminal_utilities[term_idx]
            np.testing.assert_array_almost_equal(values[node_idx], expected)

    def test_zero_sum_at_terminals(self, setup):
        """Kuhn poker is zero-sum, so terminal utilities sum to 0."""
        matrices, backend, strategy, reach = setup
        values = backward_values_simple(strategy, reach, matrices, backend)
        values = backend.asnumpy(values)

        for term_idx, node_idx in enumerate(matrices.terminal_node_indices):
            assert np.isclose(values[node_idx].sum(), 0.0)

    def test_root_value_near_zero(self, setup):
        """Under uniform strategy in symmetric game, root value should be close to 0."""
        matrices, backend, strategy, reach = setup
        values = backward_values_simple(strategy, reach, matrices, backend)
        values = backend.asnumpy(values)

        # Kuhn is not perfectly symmetric due to position advantage
        # P1 has slight disadvantage under uniform play
        # Value should be small but not necessarily exactly 0
        assert abs(values[0, 0]) < 0.5
        assert abs(values[0, 1]) < 0.5


class TestRegretMatch:
    """Test regret matching strategy update."""

    @pytest.fixture
    def backend(self):
        return get_backend('numpy')

    @pytest.fixture
    def matrices(self):
        tree = KuhnPoker().build_tree()
        return build_game_matrices(tree)

    def test_positive_regrets_normalized(self, matrices, backend):
        """Positive regrets should be normalized to probability distribution."""
        # Create regrets: first infoset has regrets [1, 3], others are 0
        regrets = np.zeros(matrices.num_infoset_actions, dtype=np.float32)
        regrets[0] = 1.0  # First action of first infoset
        regrets[1] = 3.0  # Second action of first infoset

        strategy = regret_match(regrets, matrices, backend)
        strategy = backend.asnumpy(strategy)

        # First infoset should have [0.25, 0.75]
        assert np.isclose(strategy[0], 0.25)
        assert np.isclose(strategy[1], 0.75)

    def test_negative_regrets_become_zero(self, matrices, backend):
        """Negative regrets should contribute 0 to strategy."""
        regrets = np.zeros(matrices.num_infoset_actions, dtype=np.float32)
        regrets[0] = -1.0
        regrets[1] = 2.0

        strategy = regret_match(regrets, matrices, backend)
        strategy = backend.asnumpy(strategy)

        # First infoset: [-1, 2] -> positive [0, 2] -> [0, 1]
        assert np.isclose(strategy[0], 0.0)
        assert np.isclose(strategy[1], 1.0)

    def test_all_negative_gives_uniform(self, matrices, backend):
        """All negative regrets should give uniform strategy."""
        regrets = np.full(matrices.num_infoset_actions, -1.0, dtype=np.float32)

        strategy = regret_match(regrets, matrices, backend)
        strategy = backend.asnumpy(strategy)

        # All infosets should be uniform (0.5, 0.5 for Kuhn's 2 actions)
        for h_idx in range(matrices.num_infosets):
            start = matrices.infoset_action_offsets[h_idx]
            end = matrices.infoset_action_offsets[h_idx + 1]
            num_actions = end - start
            expected = 1.0 / num_actions
            assert np.allclose(strategy[start:end], expected)

    def test_zero_regrets_gives_uniform(self, matrices, backend):
        """Zero regrets should give uniform strategy."""
        regrets = np.zeros(matrices.num_infoset_actions, dtype=np.float32)

        strategy = regret_match(regrets, matrices, backend)
        strategy = backend.asnumpy(strategy)

        for h_idx in range(matrices.num_infosets):
            start = matrices.infoset_action_offsets[h_idx]
            end = matrices.infoset_action_offsets[h_idx + 1]
            num_actions = end - start
            expected = 1.0 / num_actions
            assert np.allclose(strategy[start:end], expected)

    def test_strategy_is_valid(self, matrices, backend):
        """Any regrets should produce valid probability distributions."""
        np.random.seed(42)
        regrets = np.random.randn(matrices.num_infoset_actions).astype(np.float32)

        strategy = regret_match(regrets, matrices, backend)
        strategy = backend.asnumpy(strategy)

        # All non-negative
        assert np.all(strategy >= 0)

        # Each infoset sums to 1
        for h_idx in range(matrices.num_infosets):
            start = matrices.infoset_action_offsets[h_idx]
            end = matrices.infoset_action_offsets[h_idx + 1]
            assert np.isclose(strategy[start:end].sum(), 1.0)


class TestCounterfactualValues:
    """Test counterfactual value computation."""

    @pytest.fixture
    def setup(self):
        tree = KuhnPoker().build_tree()
        matrices = build_game_matrices(tree)
        backend = get_backend('numpy')
        strategy = uniform_strategy(matrices, backend)
        reach = forward_reach_simple(strategy, matrices, backend)
        return matrices, backend, strategy, reach

    def test_cf_values_shape(self, setup):
        matrices, backend, strategy, reach = setup
        cf_values = compute_counterfactual_values(strategy, reach, matrices, backend, player=0)
        assert cf_values.shape == (matrices.num_infoset_actions,)

    def test_both_players_cf_values(self, setup):
        """Both players should get valid CF values."""
        matrices, backend, strategy, reach = setup

        cf0 = compute_counterfactual_values(strategy, reach, matrices, backend, player=0)
        cf1 = compute_counterfactual_values(strategy, reach, matrices, backend, player=1)

        cf0 = backend.asnumpy(cf0)
        cf1 = backend.asnumpy(cf1)

        # Should have some non-zero values (unless degenerate game)
        assert not np.allclose(cf0, 0) or not np.allclose(cf1, 0)


class TestInstantRegret:
    """Test instantaneous regret computation."""

    @pytest.fixture
    def setup(self):
        tree = KuhnPoker().build_tree()
        matrices = build_game_matrices(tree)
        backend = get_backend('numpy')
        strategy = uniform_strategy(matrices, backend)
        reach = forward_reach_simple(strategy, matrices, backend)
        return matrices, backend, strategy, reach

    def test_regret_shape(self, setup):
        matrices, backend, strategy, reach = setup

        cf_values = compute_counterfactual_values(strategy, reach, matrices, backend, player=0)
        infoset_cf = compute_infoset_cf_values(cf_values, strategy, matrices, backend)
        regret = compute_instant_regret(cf_values, infoset_cf, matrices, backend)

        assert regret.shape == (matrices.num_infoset_actions,)

    def test_regret_sums_to_zero_per_infoset(self, setup):
        """Sum of regrets weighted by strategy = 0 for each infoset."""
        matrices, backend, strategy, reach = setup

        cf_values = compute_counterfactual_values(strategy, reach, matrices, backend, player=0)
        infoset_cf = compute_infoset_cf_values(cf_values, strategy, matrices, backend)
        regret = compute_instant_regret(cf_values, infoset_cf, matrices, backend)

        regret = backend.asnumpy(regret)
        strategy = backend.asnumpy(strategy)

        # For each infoset: sum of (strategy * regret) should be 0
        # Because regret[a] = cf[a] - sum(strategy[a'] * cf[a'])
        for h_idx in range(matrices.num_infosets):
            start = matrices.infoset_action_offsets[h_idx]
            end = matrices.infoset_action_offsets[h_idx + 1]
            weighted_sum = np.sum(strategy[start:end] * regret[start:end])
            assert np.isclose(weighted_sum, 0.0, atol=1e-6)


class TestCFRIteration:
    """Test a full CFR iteration."""

    @pytest.fixture
    def setup(self):
        tree = KuhnPoker().build_tree()
        matrices = build_game_matrices(tree)
        backend = get_backend('numpy')
        return matrices, backend

    def test_single_iteration(self, setup):
        """Run one CFR iteration and verify regrets change."""
        matrices, backend = setup

        # Start with uniform strategy
        strategy = uniform_strategy(matrices, backend)
        cumulative_regret = backend.zeros(matrices.num_infoset_actions)

        # Compute reach probabilities
        reach = forward_reach_simple(strategy, matrices, backend)

        # For each player, compute CF values and regrets
        for player in range(2):
            cf_values = compute_counterfactual_values(
                strategy, reach, matrices, backend, player
            )
            infoset_cf = compute_infoset_cf_values(cf_values, strategy, matrices, backend)
            instant_regret = compute_instant_regret(cf_values, infoset_cf, matrices, backend)

            # Accumulate regrets (only for this player's infosets)
            # For simplicity, we'll accumulate all (in real CFR, filter by player)
            cumulative_regret = cumulative_regret + backend.asnumpy(instant_regret)

        cumulative_regret = backend.dense_to_backend(cumulative_regret)

        # Get new strategy from regret matching
        new_strategy = regret_match(cumulative_regret, matrices, backend)

        # New strategy should differ from uniform
        strategy_np = backend.asnumpy(strategy)
        new_strategy_np = backend.asnumpy(new_strategy)

        # At least some actions should have different probabilities
        assert not np.allclose(strategy_np, new_strategy_np)


def test_ops_smoke():
    """Quick smoke test for all operations."""
    tree = KuhnPoker().build_tree()
    matrices = build_game_matrices(tree)
    backend = get_backend('numpy')

    # Uniform strategy
    strategy = uniform_strategy(matrices, backend)
    assert strategy.shape == (24,)  # 12 infosets * 2 actions

    # Forward reach
    reach = forward_reach_simple(strategy, matrices, backend)
    assert reach.shape == (61, 2)

    # Backward values
    values = backward_values_simple(strategy, reach, matrices, backend)
    assert values.shape == (61, 2)

    # CF values for player 0
    cf0 = compute_counterfactual_values(strategy, reach, matrices, backend, player=0)
    assert cf0.shape == (24,)

    # Infoset CF values
    infoset_cf = compute_infoset_cf_values(cf0, strategy, matrices, backend)
    assert infoset_cf.shape == (12,)

    # Instant regret
    regret = compute_instant_regret(cf0, infoset_cf, matrices, backend)
    assert regret.shape == (24,)

    # Regret matching
    new_strategy = regret_match(regret, matrices, backend)
    assert new_strategy.shape == (24,)

    print("\nOps smoke test passed!")
    print(f"  Strategy sample: {backend.asnumpy(strategy[:4])}")
    print(f"  Root reach: {backend.asnumpy(reach[0])}")
    print(f"  Root value: {backend.asnumpy(values[0])}")
