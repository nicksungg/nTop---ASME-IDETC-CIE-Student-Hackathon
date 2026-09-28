#!/usr/bin/env python3
"""Cap-sweep Pareto fronts for the single-point methods.

For each stress cap c in CAPS, each method solves: min mass s.t. stress <= c
(+ volume + floor constraints). The per-cap winners trace an eps-constraint-style
mass-stress front for that method — comparable to NSGA-II / qLogNEHVI fronts.
Budgets are trimmed per cap run (a front costs len(CAPS) runs).
"""
import os, pickle, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bo_bench.problem import load_cases, Case, bounds, integ, DIM
from bo_bench.baselines import run_cma
from bo_bench.scbo import run_scbo
from bo_bench.qlognei import run_qlognei
from bo_bench.tabpfn_bo import run_pfn_cei, run_pfn_ucb
from bo_bench.pareto import hypervolume, _front

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: print(*a, flush=True)

CAPS = [45, 90, 150, 220, 335, 450, 560]     # MPa; spans below-allowable trade + published 335-560 range


def de_single_cap(case, seed=0):
    """one DE run at case.cap (the inner solver of pareto_rad5.eps, single cap)"""
    case.reset(); t0 = time.time()
    def obj(Z):
        X = (Z.T if (Z.ndim == 2 and Z.shape[0] == DIM) else np.atleast_2d(Z)).copy()
        w, s, vp, vf = case.evaluate(X)
        return w + case.penalty(w, s, vp, vf)
    r = differential_evolution(obj, bounds, vectorized=True, updating="deferred",
                               integrality=integ, maxiter=100, popsize=16, seed=seed,
                               tol=1e-8, polish=False, mutation=(0.5, 1.), recombination=0.7)
    X = r.x.reshape(1, -1).copy()
    X[:, integ] = np.round(X[:, integ])
    w, s, vp, vf = case.evaluate(X)
    if not case.feasible(w, s, vp, vf)[0]:
        return None
    return dict(opt_mass=float(w[0]), pred_stress=float(s[0]), x=list(map(float, X[0])),
                wall_s=time.time() - t0, n_eval=case.n_eval)


METHODS = [
    ("de_eps",  lambda c: de_single_cap(c, seed=0)),
    ("cma_es",  lambda c: run_cma(c, seed=0, maxfevals=12_000, restarts=2)),
    ("scbo",    lambda c: run_scbo(c, seed=0, n_init=40, max_evals=700)),
    ("qlognei", lambda c: run_qlognei(c, seed=0, n_init=32, q=8, n_iter=20)),
    ("pfn_cei", lambda c: run_pfn_cei(c, seed=0, n_init=128, q=24, n_iter=18)),
    ("pfn_ucb", lambda c: run_pfn_ucb(c, seed=0, n_init=128, q=24, n_iter=18)),
]

OUT = os.path.join(ROOT, "results", "bo_bench_cap_fronts.pkl")
results = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else []
have = {(d["case"], d["method"]) for d in results}

cases = load_cases()
for ci, crow in cases.iterrows():
    P(f"\n=== case {ci} KCAS {crow.KCAS:.0f} ===")
    for name, fn in METHODS:
        if (ci, name) in have:
            continue
        pts, per_cap, wall, evals = [], [], 0.0, 0
        for cap in CAPS:
            case = Case(ci, crow, cap=cap)
            try:
                r = fn(case)
            except Exception as e:
                P(f"  {name} cap {cap}: ERROR {type(e).__name__}: {e}"); r = None
            if r is not None:
                pts.append([r["opt_mass"], r["pred_stress"]])
                per_cap.append(dict(cap=cap, mass=r["opt_mass"], stress=r["pred_stress"],
                                    x=r.get("x"), wall_s=r["wall_s"], n_eval=r["n_eval"]))
                wall += r["wall_s"]; evals += r["n_eval"]
            else:
                per_cap.append(dict(cap=cap, mass=None))
                wall += case.t_eval  # at minimum
        pts = np.array(pts) if pts else np.zeros((0, 2))
        mask = np.ones(len(pts), bool)
        fr = _front(pts[:, 0], pts[:, 1], mask) if len(pts) else pts
        fr = fr[np.argsort(fr[:, 0])] if len(fr) else fr
        d = dict(case=ci, KCAS=float(crow.KCAS), method=name, front=fr, points=pts,
                 per_cap=per_cap, hv=hypervolume(fr), wall_s=wall, n_eval=evals)
        results.append(d)
        pickle.dump(results, open(OUT, "wb"))
        P(f"  {name:9}: {len(fr)}/{len(CAPS)} front pts  HV {d['hv']:9.0f}  "
          f"{wall:6.1f}s  {evals:>7} evals")
P("\nDONE")
