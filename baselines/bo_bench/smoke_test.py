#!/usr/bin/env python3
"""Quick check that all 8 optimizers run: case 0 (KCAS 33) with tiny budgets, a few
minutes on CPU. Writes nothing. Numbers are NOT comparable to the full benchmark."""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bo_bench.problem import load_cases, Case
from bo_bench.baselines import run_de, run_cma
from bo_bench.scbo import run_scbo
from bo_bench.qlognei import run_qlognei
from bo_bench.tabpfn_bo import run_pfn_cei, run_pfn_ucb
from bo_bench.pareto import run_nsga_front, run_qlognehvi

P = lambda *a: print(*a, flush=True)
case = Case(0, load_cases().iloc[0])
P(f"case 0 | KCAS {case.KCAS:.0f} | seed mass {case.real_mass:.0f} kg")

RUNS = [
    ("de_eps",    lambda: run_de(case, seed=0)),
    ("cma_es",    lambda: run_cma(case, seed=0, maxfevals=4_000, restarts=1)),
    ("scbo",      lambda: run_scbo(case, seed=0, n_init=24, batch_size=12, max_evals=96)),
    ("qlognei",   lambda: run_qlognei(case, seed=0, n_init=16, q=4, n_iter=3)),
    ("pfn_cei",   lambda: run_pfn_cei(case, seed=0, n_init=64, q=16, n_iter=3)),
    ("pfn_ucb",   lambda: run_pfn_ucb(case, seed=0, n_init=64, q=16, n_iter=3)),
    ("nsga2",     lambda: run_nsga_front(case, seed=1, pop=40, gens=10)),
    ("qlognehvi", lambda: run_qlognehvi(case, seed=0, n_init=16, q=2, n_iter=3, device="cpu")),
]
ok = 0
for name, fn in RUNS:
    t = time.time()
    try:
        r = fn()
    except Exception as e:
        P(f"  {name:10}: FAIL {type(e).__name__}: {e}"); continue
    ok += 1
    if r is None:
        P(f"  {name:10}: ran, no feasible design at this budget ({time.time()-t:.0f}s)")
    elif "front" in r:
        P(f"  {name:10}: HV {r['hv']:9.0f}  front {len(r['front'])} pts  ({time.time()-t:.0f}s)")
    else:
        P(f"  {name:10}: {r['opt_mass']:6.1f} kg  stress {r['pred_stress']:5.0f} MPa  ({time.time()-t:.0f}s)")
P(f"\n{ok}/{len(RUNS)} optimizers ran")
sys.exit(0 if ok == len(RUNS) else 1)
