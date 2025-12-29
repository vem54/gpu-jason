"""
Basic import tests to verify package structure.

Run with: pytest tests/test_imports.py -v
"""

import pytest


class TestPackageImports:
    """Test that all package modules can be imported."""

    def test_import_main_package(self):
        """Main package should be importable."""
        import gpu_poker_cfr
        assert gpu_poker_cfr.__version__ == "0.1.0"

    def test_import_games(self):
        """Games layer should be importable."""
        import gpu_poker_cfr.games
        assert gpu_poker_cfr.games is not None

    def test_import_matrix(self):
        """Matrix layer should be importable."""
        import gpu_poker_cfr.matrix
        assert gpu_poker_cfr.matrix is not None

    def test_import_engine(self):
        """Engine layer should be importable."""
        import gpu_poker_cfr.engine
        assert gpu_poker_cfr.engine is not None

    def test_import_solvers(self):
        """Solvers layer should be importable."""
        import gpu_poker_cfr.solvers
        assert gpu_poker_cfr.solvers is not None

    def test_import_testing(self):
        """Testing layer should be importable."""
        import gpu_poker_cfr.testing
        assert gpu_poker_cfr.testing is not None


class TestDependencyAvailability:
    """Test that required dependencies are available."""

    def test_numpy_available(self):
        """NumPy should be installed."""
        import numpy as np
        assert np.__version__ is not None

    def test_scipy_available(self):
        """SciPy should be installed."""
        import scipy
        assert scipy.__version__ is not None

    @pytest.mark.skipif(True, reason="CuPy requires CUDA GPU")
    def test_cupy_available(self):
        """CuPy should be installed (skip if no GPU)."""
        try:
            import cupy as cp
            assert cp.__version__ is not None
        except ImportError:
            pytest.skip("CuPy not installed or no CUDA GPU available")
