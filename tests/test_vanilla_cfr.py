"""
Tests for Vanilla CFR solver.

Run with: pytest tests/test_vanilla_cfr.py -v
"""

import pytest
import numpy as np

from gpu_poker_cfr.games.kuhn import KuhnPoker
from gpu_poker_cfr.solvers.vanilla import VanillaCFR


class TestVanillaCFRBasic:
    """Basic tests for VanillaCFR solver."""

    def test_initialization(self):
        """Solver should initialize without error."""
        solver = VanillaCFR(KuhnPoker(), backend='numpy')
        assert solver.iterations == 0
        assert solver.matrices.num_infosets == 12

    def test_single_iteration(self):
        """Should complete a single iteration."""
        solver = VanillaCFR(KuhnPoker())
        solver.iterate(1)
        assert solver.iterations == 1

    def test_multiple_iterations(self):
        """Should complete multiple iterations."""
        solver = VanillaCFR(KuhnPoker())
        solver.iterate(10)
        assert solver.iterations == 10

    def test_solve(self):
        """solve() should run iterations."""
        solver = VanillaCFR(KuhnPoker())
        solver.solve(iterations=100)
        assert solver.iterations == 100


class TestVanillaCFRStrategy:
    """Tests for strategy computation."""

    @pytest.fixture
    def solver(self):
        solver = VanillaCFR(KuhnPoker())
        solver.solve(iterations=100)
        return solver

    def test_current_strategy_shape(self, solver):
        """Current strategy should have correct shape."""
        strategy = solver.current_strategy
        assert strategy.shape == (24,)  # 12 infosets × 2 actions

    def test_average_strategy_shape(self, solver):
        """Average strategy should have correct shape."""
        strategy = solver.average_strategy
        assert strategy.shape == (24,)

    def test_strategy_is_valid_distribution(self, solver):
        """Strategy should be valid probability distribution per infoset."""
        strategy = solver.average_strategy

        for h_idx in range(solver.matrices.num_infosets):
            start = solver.matrices.infoset_action_offsets[h_idx]
            end = solver.matrices.infoset_action_offsets[h_idx + 1]

            probs = strategy[start:end]
            assert np.all(probs >= 0), f"Negative probability at infoset {h_idx}"
            assert np.isclose(probs.sum(), 1.0), f"Probabilities don't sum to 1 at infoset {h_idx}"

    def test_get_strategy_for_infoset(self, solver):
        """Should get strategy for specific infoset."""
        for h_idx in range(solver.matrices.num_infosets):
            strategy = solver.get_strategy_for_infoset(h_idx)
            assert len(strategy) == 2  # Kuhn has 2 actions per infoset
            assert np.all(strategy >= 0)
            assert np.isclose(strategy.sum(), 1.0)


class TestVanillaCFRConvergence:
    """Tests for CFR convergence properties."""

    def test_exploitability_decreases(self):
        """Exploitability should decrease over iterations."""
        solver = VanillaCFR(KuhnPoker())

        # Measure exploitability at different iteration counts
        exploitabilities = []

        for target_iters in [10, 100, 500]:
            solver.iterate(target_iters - solver.iterations)
            expl = solver.exploitability()
            exploitabilities.append(expl)

        # Exploitability should generally decrease
        # Allow some noise but trend should be downward
        assert exploitabilities[-1] < exploitabilities[0], \
            f"Exploitability didn't decrease: {exploitabilities}"

    def test_converges_to_low_exploitability(self):
        """Should converge to near-zero exploitability."""
        solver = VanillaCFR(KuhnPoker())
        solver.solve(iterations=1000)

        expl = solver.exploitability()
        # Kuhn Nash has exploitability 0
        # After 1000 iterations, should be < 0.1
        assert expl < 0.1, f"Exploitability {expl} too high after 1000 iterations"

    @pytest.mark.slow
    def test_high_iteration_convergence(self):
        """Long run should converge to very low exploitability."""
        solver = VanillaCFR(KuhnPoker())
        solver.solve(iterations=10000)

        expl = solver.exploitability()
        assert expl < 0.01, f"Exploitability {expl} too high after 10000 iterations"


class TestKuhnNashEquilibrium:
    """Tests for Kuhn poker Nash equilibrium properties."""

    @pytest.fixture
    def converged_solver(self):
        solver = VanillaCFR(KuhnPoker())
        solver.solve(iterations=10000)  # More iterations for convergence
        return solver

    def test_p1_jack_betting_frequency(self, converged_solver):
        """P1 with Jack should bet ~1/3 as a bluff (Nash)."""
        solver = converged_solver

        # Find P1 Jack infoset (first action for P1 with J)
        for h_idx in range(solver.matrices.num_infosets):
            name = solver.get_infoset_name(h_idx)
            if name == "J:":
                strategy = solver.get_strategy_for_infoset(h_idx)
                # strategy[0] = check, strategy[1] = bet
                bet_freq = strategy[1]
                # Nash: bet with probability α ≈ 1/3
                # Wide tolerance due to CFR's slow convergence
                assert 0.1 < bet_freq < 0.6, f"J: bet frequency {bet_freq} outside expected range"
                break

    def test_p1_king_always_bets(self, converged_solver):
        """P1 with King should bet frequently (Nash)."""
        solver = converged_solver

        for h_idx in range(solver.matrices.num_infosets):
            name = solver.get_infoset_name(h_idx)
            if name == "K:":
                strategy = solver.get_strategy_for_infoset(h_idx)
                bet_freq = strategy[1]
                # Nash: bet with K (CFR converges slowly, allow wide tolerance)
                assert bet_freq > 0.5, f"K: bet frequency {bet_freq} too low"
                break

    def test_p2_king_always_calls(self, converged_solver):
        """P2 with King facing bet should always call (Nash)."""
        solver = converged_solver

        for h_idx in range(solver.matrices.num_infosets):
            name = solver.get_infoset_name(h_idx)
            if name == "K:b":
                strategy = solver.get_strategy_for_infoset(h_idx)
                # strategy[0] = fold, strategy[1] = call
                call_freq = strategy[1]
                assert call_freq > 0.8, f"K:b call frequency {call_freq} too low"
                break

    def test_p2_jack_always_folds(self, converged_solver):
        """P2 with Jack facing bet should always fold (Nash)."""
        solver = converged_solver

        for h_idx in range(solver.matrices.num_infosets):
            name = solver.get_infoset_name(h_idx)
            if name == "J:b":
                strategy = solver.get_strategy_for_infoset(h_idx)
                # strategy[0] = fold, strategy[1] = call
                fold_freq = strategy[0]
                assert fold_freq > 0.8, f"J:b fold frequency {fold_freq} too low"
                break


class TestVanillaCFROutput:
    """Tests for solver output methods."""

    def test_print_strategy_runs(self, capsys):
        """print_strategy should run without error."""
        solver = VanillaCFR(KuhnPoker())
        solver.solve(iterations=10)
        solver.print_strategy()

        captured = capsys.readouterr()
        assert "Average Strategy" in captured.out

    def test_get_infoset_name(self):
        """Should return readable infoset names."""
        solver = VanillaCFR(KuhnPoker())

        # Check some expected names exist
        names = [solver.get_infoset_name(i) for i in range(solver.matrices.num_infosets)]

        assert "J:" in names or any("J" in n for n in names)
        assert "Q:" in names or any("Q" in n for n in names)
        assert "K:" in names or any("K" in n for n in names)


def test_vanilla_cfr_smoke():
    """Quick smoke test for VanillaCFR."""
    solver = VanillaCFR(KuhnPoker(), backend='numpy')
    solver.solve(iterations=100)

    # Basic checks
    assert solver.iterations == 100
    assert solver.average_strategy.shape == (24,)
    assert solver.exploitability() > 0  # Should be positive (not exactly Nash)

    print(f"\nSmoke test results:")
    print(f"  Iterations: {solver.iterations}")
    print(f"  Exploitability: {solver.exploitability():.6f}")


@pytest.mark.slow
def test_vanilla_cfr_full_convergence():
    """Full convergence test (marked slow)."""
    solver = VanillaCFR(KuhnPoker())
    solver.solve(iterations=10000)

    expl = solver.exploitability()
    print(f"\nFull convergence test:")
    print(f"  Iterations: {solver.iterations}")
    print(f"  Exploitability: {expl:.6f}")

    solver.print_strategy()

    assert expl < 0.01
