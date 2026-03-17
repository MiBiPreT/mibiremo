"""Benchmark: compare speed of pure-Python Sauly'ev algotithm
vs the Numba-optimized version in SemiLagSolver.saulyev_solver_alt()
"""

import time
import warnings

import numpy as np
from mibiremo.semilagsolver import SemiLagSolver

warnings.filterwarnings("ignore")


class _PurePythonSolver(SemiLagSolver):
    """pure-Python Saulyev implementation for benchmarking purposes"""

    def saulyev_solver_alt(self, c_bound):
        theta = self.d * self.dt / (self.dx**2)
        c_init = self.C.copy()
        clr = self.C.copy()
        crl = self.C.copy()
        inv = 1.0 / (1.0 + theta)

        for i in range(len(clr)):
            sola = theta * c_bound if i == 0 else theta * clr[i - 1]
            solb = (1 - theta) * c_init[i]
            solc = theta * c_init[i + 1] if i < len(clr) - 1 else theta * c_init[i]
            clr[i] = (sola + solb + solc) * inv

        for i in range(len(crl) - 1, -1, -1):
            sola = theta * clr[-1] if i == len(crl) - 1 else theta * crl[i + 1]
            solb = (1 - theta) * c_init[i]
            solc = theta * c_init[i - 1] if i > 0 else theta * c_init[i]
            crl[i] = (sola + solb + solc) * inv

        self.C = (clr + crl) / 2


def make_solver(solver, n):
    x = np.linspace(0, 10, n)
    c = np.exp(-((x - 2) ** 2) / 0.5)
    return solver(x, c, 0.3, 0.01, 0.1)


def benchmark(fn, n_warmup=3, n_runs=100):
    # First run a few warmup iterations
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    # Then run the actual benchmark
    for _ in range(n_runs):
        fn()
    return (time.perf_counter() - t0) / n_runs * 1e6  # µs


def run_all(n_grid, n_runs=500):
    s_py = make_solver(_PurePythonSolver, n_grid)
    s_opt = make_solver(SemiLagSolver, n_grid)
    c_ini = s_py.C.copy()

    def bench_py():
        s_py.C = c_ini.copy()
        s_py.saulyev_solver_alt(1.0)

    def bench_opt():
        s_opt.C = c_ini.copy()
        s_opt.saulyev_solver_alt(1.0)

    t_py = benchmark(bench_py, n_runs=n_runs)
    t_def = benchmark(bench_opt, n_runs=n_runs)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  saulyev_solver_alt  grid={n_grid}  runs={n_runs}")
    print(sep)
    print(f"  {'Strategy':<36s}  {'µs/call':>8s}  {'speedup':>7s}")
    print(f"  {'-' * 36}  {'-' * 8}  {'-' * 7}")
    print(f"  {'pure-Python':<36s}  {t_py:8.2f}  {'1.00x':>7s}")
    print(f"  {'Numba JIT':<36s}  {t_def:8.2f}  {t_py / t_def:7.2f}x")
    print(f"{sep}\n")


if __name__ == "__main__":
    # Run benchmarks for various grid sizes and number of runs
    run_all(n_grid=200, n_runs=1000)
    run_all(n_grid=500, n_runs=500)
    run_all(n_grid=2000, n_runs=200)
    run_all(n_grid=10000, n_runs=50)
