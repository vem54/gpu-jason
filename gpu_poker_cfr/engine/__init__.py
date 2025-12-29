"""
Compute engine layer (Layer 3).

This layer provides GPU/CPU matrix operations for CFR.
It may only import from: gpu_poker_cfr.games, gpu_poker_cfr.matrix
"""

from gpu_poker_cfr.engine.backend import (
    get_backend,
    to_numpy,
    is_cupy_available,
    Backend,
)

from gpu_poker_cfr.engine.ops import (
    forward_reach_simple,
    backward_values_simple,
    compute_counterfactual_values,
    compute_infoset_cf_values,
    compute_instant_regret,
    regret_match,
    uniform_strategy,
)

__all__ = [
    'get_backend',
    'to_numpy',
    'is_cupy_available',
    'Backend',
    'forward_reach_simple',
    'backward_values_simple',
    'compute_counterfactual_values',
    'compute_infoset_cf_values',
    'compute_instant_regret',
    'regret_match',
    'uniform_strategy',
]
