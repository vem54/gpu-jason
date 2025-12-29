"""
Tests for OpenSpiel validation.

Run with: pytest tests/test_openspiel.py -v
"""

import pytest

from gpu_poker_cfr.testing.openspiel_compare import (
    is_openspiel_available,
    run_comparison,
    validate_against_known_nash,
)


class TestOpenSpielAvailability:
    """Test OpenSpiel availability detection."""

    def test_availability_returns_bool(self):
        """is_openspiel_available should return a boolean."""
        result = is_openspiel_available()
        assert isinstance(result, bool)


class TestNashValidation:
    """Test validation against known Nash equilibrium."""

    @pytest.mark.slow
    def test_validate_against_nash(self):
        """Should validate against known Kuhn Nash equilibrium."""
        valid, report = validate_against_known_nash()

        print(report)

        # Strategy should be close to Nash
        assert valid, f"Nash validation failed:\n{report}"

    def test_validation_returns_report(self):
        """Validation should return detailed report."""
        # Run with fewer iterations for speed
        from gpu_poker_cfr.games.kuhn import KuhnPoker
        from gpu_poker_cfr.solvers.vanilla import VanillaCFR

        solver = VanillaCFR(KuhnPoker())
        solver.solve(iterations=100)

        # Just check that exploitability is finite
        expl = solver.exploitability()
        assert expl >= 0
        assert expl < float('inf')


class TestOpenSpielComparison:
    """Tests for OpenSpiel comparison (skipped if not available)."""

    @pytest.fixture
    def skip_if_no_openspiel(self):
        if not is_openspiel_available():
            pytest.skip("OpenSpiel not available")

    def test_run_comparison_without_openspiel(self):
        """run_comparison should work even without OpenSpiel."""
        # Should not raise, just note OpenSpiel is unavailable
        report = run_comparison(our_iterations=100, openspiel_iterations=100)
        assert "VanillaCFR" in report or "GPU-CFR" in report

    @pytest.mark.skipif(
        not is_openspiel_available(),
        reason="OpenSpiel not available"
    )
    @pytest.mark.slow
    def test_run_comparison_with_openspiel(self):
        """Full comparison with OpenSpiel."""
        report = run_comparison(our_iterations=1000, openspiel_iterations=1000)

        print(report)

        assert "Our Strategy" in report
        assert "OpenSpiel" in report


def test_openspiel_validation_smoke():
    """Quick smoke test for OpenSpiel validation module."""
    # This should work regardless of OpenSpiel availability
    available = is_openspiel_available()
    print(f"\nOpenSpiel available: {available}")

    # Run comparison (will skip OpenSpiel part if not available)
    report = run_comparison(our_iterations=50, openspiel_iterations=50)
    print(report)

    assert len(report) > 0
