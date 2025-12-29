"""
CFR solver algorithms layer (Layer 4 - highest).

This layer implements CFR variants (Vanilla, CFR+, Partial Vector CFR).
It may import from: gpu_poker_cfr.games, gpu_poker_cfr.matrix, gpu_poker_cfr.engine
"""

from gpu_poker_cfr.solvers.vanilla import VanillaCFR

__all__ = ['VanillaCFR']
