"""
Tests for CFR+ solver.

Run with: pytest tests/test_cfr_plus.py -v
"""

import pytest
import numpy as np

from gpu_poker_cfr.games.kuhn import KuhnPoker
from gpu_poker_cfr.solvers.cfr_plus import CFRPlus
from gpu_poker_cfr.solvers.vanilla import VanillaCFR


# Game value at Nash equilibrium (same for all Nash equilibria)
NASH_GAME_VALUE = -1.0 / 18.0


def get_strategy_dict(solver, infoset_name):
    """Get strategy as dict {action_name: probability} for named infoset."""
    for h_idx in range(solver.matrices.num_infosets):
        name = solver.get_infoset_name(h_idx)
        if name == infoset_name:
            strategy = solver.get_strategy_for_infoset(h_idx)
            infoset = solver._tree.infosets[h_idx]
            return {infoset.actions[i].name: strategy[i] for i in range(len(infoset.actions))}
    return None


class TestCFRPlusBasic:
    """Basic tests for CFR+ solver."""

    def test_initialization(self):
        """Solver should initialize without error."""
        solver = CFRPlus(KuhnPoker(), backend='numpy')
        assert solver.iterations == 0
        assert solver.matrices.num_infosets == 12

    def test_single_iteration(self):
        """Should complete a single iteration."""
        solver = CFRPlus(KuhnPoker())
        solver.iterate(1)
        assert solver.iterations == 1

    def test_solve(self):
        """solve() should run iterations."""
        solver = CFRPlus(KuhnPoker())
        solver.solve(iterations=100)
        assert solver.iterations == 100


class TestCFRPlusConvergence:
    """Tests for CFR+ convergence properties."""

    @pytest.fixture(scope="class")
    def converged_solver(self):
        """Run CFR+ for enough iterations to converge."""
        solver = CFRPlus(KuhnPoker(), backend='numpy')
        solver.solve(iterations=10000)
        return solver

    def test_exploitability_low(self, converged_solver):
        """Exploitability should be low after convergence."""
        expl = converged_solver.exploitability()
        # CFR+ should achieve < 0.01 in 10k iterations
        assert expl < 0.01, f"Exploitability {expl} too high"

    def test_game_value(self, converged_solver):
        """Game value should be -1/18."""
        from gpu_poker_cfr.engine.ops import forward_reach_simple, backward_values_simple

        solver = converged_solver
        avg_strategy = solver.average_strategy
        avg_strategy_backend = solver.backend.dense_to_backend(avg_strategy)

        reach = forward_reach_simple(avg_strategy_backend, solver.matrices, solver.backend)
        values = backward_values_simple(avg_strategy_backend, reach, solver.matrices, solver.backend)
        values_np = solver.backend.asnumpy(values)

        p1_value = values_np[0, 0]

        assert abs(p1_value - NASH_GAME_VALUE) < 0.01, \
            f"Game value {p1_value} != expected {NASH_GAME_VALUE}"

    def test_p1_alpha_relationship(self, converged_solver):
        """P1 K: bet should be approximately 3 × P1 J: bet."""
        solver = converged_solver

        p1_j_bet = get_strategy_dict(solver, "J:")["b"]
        p1_k_bet = get_strategy_dict(solver, "K:")["b"]

        ratio = p1_k_bet / p1_j_bet if p1_j_bet > 0.01 else float('inf')

        assert 2.5 < ratio < 3.5, \
            f"P1 K/J ratio {ratio:.2f} not close to 3.0"

    def test_dominated_strategies(self, converged_solver):
        """Dominated strategies should converge to pure values."""
        solver = converged_solver

        # J facing bet should always fold
        assert get_strategy_dict(solver, "J:b")["c"] < 0.05

        # K facing bet should always call
        assert get_strategy_dict(solver, "K:b")["c"] > 0.95

        # K after check should always bet
        assert get_strategy_dict(solver, "K:c")["b"] > 0.95

        # K facing check-bet should always call
        assert get_strategy_dict(solver, "K:c,b")["c"] > 0.95

        # J facing check-bet should always fold
        assert get_strategy_dict(solver, "J:c,b")["c"] < 0.05


class TestCFRPlusFasterThanVanilla:
    """Test that CFR+ converges faster than Vanilla CFR."""

    def test_faster_convergence(self):
        """CFR+ should have lower exploitability than Vanilla at same iterations."""
        iterations = 5000

        cfr_plus = CFRPlus(KuhnPoker(), backend='numpy')
        cfr_plus.solve(iterations=iterations)

        vanilla = VanillaCFR(KuhnPoker(), backend='numpy')
        vanilla.solve(iterations=iterations)

        cfr_plus_expl = cfr_plus.exploitability()
        vanilla_expl = vanilla.exploitability()

        # CFR+ should be at least 1.5x better
        assert cfr_plus_expl < vanilla_expl, \
            f"CFR+ ({cfr_plus_expl}) not faster than Vanilla ({vanilla_expl})"

        speedup = vanilla_expl / cfr_plus_expl
        print(f"\nCFR+ speedup: {speedup:.1f}x lower exploitability")


def test_cfr_plus_smoke():
    """Quick smoke test for CFR+."""
    solver = CFRPlus(KuhnPoker(), backend='numpy')
    solver.solve(iterations=1000)

    assert solver.iterations == 1000
    assert solver.average_strategy.shape == (24,)
    assert solver.exploitability() < 0.02

    print(f"\nSmoke test results:")
    print(f"  Iterations: {solver.iterations}")
    print(f"  Exploitability: {solver.exploitability():.6f}")
