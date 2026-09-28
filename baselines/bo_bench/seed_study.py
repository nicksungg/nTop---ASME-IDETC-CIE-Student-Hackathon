#!/usr/bin/env python3
"""Seed replicates on the discriminating case (case 0, KCAS 33 — mass floor not
binding) to check robustness of the method ranking. 3 seeds per method."""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bo_bench.problem import load_cases, Case
from bo_bench.baselines import run_de, run_cma
from bo_bench.scbo import run_scbo
from bo_bench.qlognei import run_qlognei
from bo_bench.tabpfn_bo import run_pfn_cei, run_pfn_ucb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [0, 1, 2]
P = lambda *a: print(*a, flush=True)

cases = load_cases()
rows = []
for seed in SEEDS:
    case = Case(0, cases.iloc[0])
    for name, fn in [
        ("de_eps",  lambda c, s=seed: run_de(c, seed=s)),
        ("cma_es",  lambda c, s=seed: run_cma(c, seed=s, maxfevals=40_000)),
        ("scbo",    lambda c, s=seed: run_scbo(c, seed=s, max_evals=1500)),
        ("qlognei", lambda c, s=seed: run_qlognei(c, seed=s, q=8, n_iter=44)),
        ("pfn_cei", lambda c, s=seed: run_pfn_cei(c, seed=s, n_iter=34)),
        ("pfn_ucb", lambda c, s=seed: run_pfn_ucb(c, seed=s, n_iter=34)),
    ]:
        t = time.time()
        try:
            r = fn(case)
        except Exception as e:
            P(f"seed {seed} {name}: ERROR {type(e).__name__}: {e}"); r = None
        if r is None:
            P(f"seed {seed} {name}: no feasible"); continue
        r.pop("x", None); r["seed"] = seed
        rows.append(r)
        P(f"seed {seed} {name:9}: {r['opt_mass']:6.2f} kg  {r['wall_s']:6.1f}s  {r['n_eval']:>7} evals")
        pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "bo_bench_seeds.csv"), index=False)
P("DONE")
