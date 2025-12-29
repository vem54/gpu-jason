"""Benchmark GPU CFR v4 (fused kernel) vs v3 and CPU."""

import time
import numpy as np

# GPU
import cupy as cp
from gpu_poker_cfr.solvers.gpu_river_cfr_v4 import GPURiverCFRv4
from gpu_poker_cfr.solvers.gpu_river_cfr_v3 import GPURiverCFRv3
from gpu_poker_cfr.solvers.semi_vector_river import SemiVectorRiver
from gpu_poker_cfr.games.river_toy import RiverMicro, RiverToyMini


def benchmark_solver(solver_class, game, name, iterations=100, warmup=10):
    """Benchmark a solver."""
    solver = solver_class(game)

    # Warmup
    solver.iterate(warmup)

    # Synchronize GPU if applicable
    if hasattr(cp.cuda, 'Stream'):
        cp.cuda.Stream.null.synchronize()

    # Benchmark
    start = time.perf_counter()
    solver.iterate(iterations)

    if hasattr(cp.cuda, 'Stream'):
        cp.cuda.Stream.null.synchronize()

    elapsed = time.perf_counter() - start
    iter_per_sec = iterations / elapsed

    print(f"{name}: {iter_per_sec:.1f} iter/s ({elapsed:.3f}s for {iterations} iters)")
    return iter_per_sec


def main():
    print("=" * 60)
    print("GPU CFR v4 Benchmark (Fused CUDA Kernel)")
    print("=" * 60)

    # RiverMicro - 24 deals
    print("\n--- RiverMicro (24 deals) ---")
    game = RiverMicro()
    print(f"Deals: {game.num_deals}")

    try:
        gpu_v4 = benchmark_solver(GPURiverCFRv4, game, "GPU v4 (fused)", iterations=1000)
    except Exception as e:
        print(f"GPU v4 error: {e}")
        gpu_v4 = 0

    try:
        gpu_v3 = benchmark_solver(GPURiverCFRv3, game, "GPU v3 (scatter)", iterations=1000)
    except Exception as e:
        print(f"GPU v3 error: {e}")
        gpu_v3 = 0

    cpu = benchmark_solver(SemiVectorRiver, game, "CPU (Numba)", iterations=1000)

    if gpu_v4 > 0:
        print(f"\nv4 vs CPU: {gpu_v4/cpu:.2f}x")
    if gpu_v3 > 0 and gpu_v4 > 0:
        print(f"v4 vs v3: {gpu_v4/gpu_v3:.2f}x")

    # RiverToyMini - 465K deals
    print("\n--- RiverToyMini (465K deals) ---")
    game = RiverToyMini()
    print(f"Deals: {game.num_deals}")

    try:
        gpu_v4 = benchmark_solver(GPURiverCFRv4, game, "GPU v4 (fused)", iterations=100)
    except Exception as e:
        print(f"GPU v4 error: {e}")
        import traceback
        traceback.print_exc()
        gpu_v4 = 0

    try:
        gpu_v3 = benchmark_solver(GPURiverCFRv3, game, "GPU v3 (scatter)", iterations=20)
    except Exception as e:
        print(f"GPU v3 error: {e}")
        gpu_v3 = 0

    cpu = benchmark_solver(SemiVectorRiver, game, "CPU (Numba)", iterations=100)

    if gpu_v4 > 0:
        print(f"\nv4 vs CPU: {gpu_v4/cpu:.2f}x")


if __name__ == "__main__":
    main()
