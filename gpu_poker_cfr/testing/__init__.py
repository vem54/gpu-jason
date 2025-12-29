"""
Testing infrastructure.

This module provides golden output management, numerical equivalence checks,
and snapshot utilities. It may import from any layer (test-only code).
"""

from gpu_poker_cfr.testing.openspiel_compare import (
    is_openspiel_available,
    run_comparison,
    validate_against_known_nash,
)

__all__ = [
    'is_openspiel_available',
    'run_comparison',
    'validate_against_known_nash',
]
