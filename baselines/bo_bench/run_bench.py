#!/usr/bin/env python3
"""Full optimizer benchmark: 6 operating points x 6 single-objective methods
+ 2 Pareto-front methods. Writes results/bo_bench_results.csv, bo_bench_fronts.pkl,
bo_bench_plans.json (winning designs in nTop schema order for FE validation)."""
import json, os, pickle, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bo_bench.problem import load_cases, Case
from bo_bench.baselines import run_de, run_cma
from bo_bench.scbo import run_scbo
from bo_bench.qlognei import run_qlognei
from bo_bench.tabpfn_bo import run_pfn_cei, run_pfn_ucb
from bo_bench.pareto import run_nsga_front, run_qlognehvi

P = lambda *a: print(*a, flush=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

METHODS = [
    ("de_eps",   lambda c: run_de(c, seed=0)),
    ("cma_es",   lambda c: run_cma(c, seed=0, maxfevals=40_000)),
    ("scbo",     lambda c: run_scbo(c, seed=0, max_evals=1500)),
    ("qlognei",  lambda c: run_qlognei(c, seed=0, q=8, n_iter=44)),
    ("pfn_cei",  lambda c: run_pfn_cei(c, seed=0, n_iter=34)),
    ("pfn_ucb",  lambda c: run_pfn_ucb(c, seed=0, n_iter=34)),
]

def main():
    cases = load_cases()
    rows, fronts, plans = [], [], {}
    for ci, crow in cases.iterrows():
        case = Case(ci, crow)
        P(f"\n=== case {ci} KCAS {case.KCAS:.0f} | seed mass {case.real_mass:.0f} kg ===")
        for name, fn in METHODS:
            t = time.time()
            try:
                r = fn(case)
            except Exception as e:
                P(f"  {name:9}: ERROR {type(e).__name__}: {e}"); r = None
            if r is None:
                P(f"  {name:9}: no feasible design ({time.time()-t:.0f}s)")
                rows.append(dict(case=ci, KCAS=case.KCAS, real_mass=case.real_mass,
                                 method=name, opt_mass=np.nan, feasible=False,
                                 wall_s=time.time() - t, n_eval=case.n_eval))
                continue
            x = r.pop("x", None)
            if x is not None:
                plans[f"{name}_case{ci}"] = x
            rows.append(r)
            P(f"  {name:9}: {r['opt_mass']:6.1f} kg ({r['pct_lighter']:+.1f}%)  "
              f"stress {r['pred_stress']:5.0f} MPa  | {r['wall_s']:6.1f}s  {r['n_eval']:>7} evals")
            pd.DataFrame(rows).to_csv(os.path.join(RES, "bo_bench_results.csv"), index=False)
        # Pareto front methods
        for fn in (lambda: run_nsga_front(case, seed=1),
                   lambda: run_qlognehvi(case, seed=0, q=8, n_iter=32)):
            try:
                fr = fn()
            except Exception as e:
                P(f"  front: ERROR {type(e).__name__}: {e}"); fr = None
            if fr:
                fronts.append(fr)
                P(f"  {fr['method']:9}: HV {fr['hv']:9.0f}  front {len(fr['front'])} pts  "
                  f"| {fr['wall_s']:6.1f}s  {fr['n_eval']:>7} evals")
                with open(os.path.join(RES, "bo_bench_fronts.pkl"), "wb") as f:
                    pickle.dump(fronts, f)
        json.dump(plans, open(os.path.join(RES, "bo_bench_plans.json"), "w"), indent=1)
    P("\nDONE")

if __name__ == "__main__":
    main()
