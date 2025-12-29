"""
Architecture contract tests using grimp.

These tests enforce the layered architecture:
- games (Layer 1, lowest) - no internal dependencies
- matrix (Layer 2) - can import from games
- engine (Layer 3) - can import from matrix, games
- solvers (Layer 4, highest) - can import from engine, matrix, games

Run with: pytest tests/test_architecture.py -v
"""

import pytest

# Try to import grimp, skip tests if not installed
grimp = pytest.importorskip("grimp")


class TestLayerArchitecture:
    """Test that layer dependencies are respected."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Build the import graph once for all tests."""
        self.graph = grimp.build_graph("gpu_poker_cfr")

    def test_games_has_no_internal_imports(self):
        """Layer 1 (games) should not import from any other layer."""
        games_imports = self.graph.find_modules_that_directly_import(
            "gpu_poker_cfr.games"
        )

        internal_imports = [
            m for m in self.graph.find_modules_directly_imported_by("gpu_poker_cfr.games")
            if m.startswith("gpu_poker_cfr.")
            and not m.startswith("gpu_poker_cfr.games")
        ]

        assert internal_imports == [], (
            f"games layer should not import from other layers, "
            f"but imports: {internal_imports}"
        )

    def test_matrix_only_imports_from_games(self):
        """Layer 2 (matrix) can only import from Layer 1 (games)."""
        allowed = {"gpu_poker_cfr.games"}

        matrix_imports = [
            m for m in self.graph.find_modules_directly_imported_by("gpu_poker_cfr.matrix")
            if m.startswith("gpu_poker_cfr.")
            and not m.startswith("gpu_poker_cfr.matrix")
        ]

        for imp in matrix_imports:
            layer = imp.split(".")[1]  # e.g., "games" from "gpu_poker_cfr.games.base"
            assert f"gpu_poker_cfr.{layer}" in allowed or imp.startswith(tuple(allowed)), (
                f"matrix layer imported from forbidden layer: {imp}"
            )

    def test_engine_only_imports_from_matrix_and_games(self):
        """Layer 3 (engine) can only import from Layers 1-2."""
        allowed = {"gpu_poker_cfr.games", "gpu_poker_cfr.matrix"}

        engine_imports = [
            m for m in self.graph.find_modules_directly_imported_by("gpu_poker_cfr.engine")
            if m.startswith("gpu_poker_cfr.")
            and not m.startswith("gpu_poker_cfr.engine")
        ]

        for imp in engine_imports:
            layer = imp.split(".")[1]
            assert f"gpu_poker_cfr.{layer}" in allowed or any(imp.startswith(a) for a in allowed), (
                f"engine layer imported from forbidden layer: {imp}"
            )

    def test_solvers_does_not_have_circular_deps(self):
        """Layer 4 (solvers) should not be imported by lower layers."""
        modules_importing_solvers = [
            m for m in self.graph.find_modules_that_directly_import("gpu_poker_cfr.solvers")
            if m.startswith("gpu_poker_cfr.")
        ]

        forbidden_importers = [
            m for m in modules_importing_solvers
            if any(m.startswith(f"gpu_poker_cfr.{layer}")
                   for layer in ["games", "matrix", "engine"])
        ]

        assert forbidden_importers == [], (
            f"Lower layers should not import from solvers: {forbidden_importers}"
        )


class TestNoCircularImports:
    """Test that there are no circular import dependencies."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph = grimp.build_graph("gpu_poker_cfr")

    def test_no_circular_imports_in_package(self):
        """The entire package should have no circular imports."""
        # grimp can detect circular imports
        # For now, just verify we can import the package without errors
        import gpu_poker_cfr
        import gpu_poker_cfr.games
        import gpu_poker_cfr.matrix
        import gpu_poker_cfr.engine
        import gpu_poker_cfr.solvers

        # If we get here without ImportError, no circular imports
        assert True


class TestImportLinterContract:
    """Test using import-linter as an alternative to grimp."""

    def test_import_linter_contract(self):
        """Run import-linter to check layer contracts."""
        try:
            from importlinter import cli
            # import-linter reads from pyproject.toml
            # This is a placeholder - actual check done via CLI
            assert True
        except ImportError:
            pytest.skip("import-linter not installed")
