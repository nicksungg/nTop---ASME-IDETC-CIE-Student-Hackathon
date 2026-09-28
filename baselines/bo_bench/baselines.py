"""Baseline optimizers: DE cap-sweep (identical to pareto_rad5.eps) and CMA-ES."""
import time
import numpy as np
from scipy.optimize import differential_evolution

from .problem import (Case, bounds, integ, ALLOW, MASS_FLOOR, LB, UB, DIM,
                      unit_to_x, result_row)

CAPS = [220, 280, 335, 420, 560]


def run_de(case: Case, seed=0):
    """Exact replica of pareto_rad5.eps(): DE min-mass under a swept stress cap."""
    case.reset(); t0 = time.time(); best = None
    for c in CAPS:
        def obj(Z):
            X = (Z.T if (Z.ndim == 2 and Z.shape[0] == DIM) else np.atleast_2d(Z)).copy()
            w, s, vp, vf = case.evaluate(X)
            return w + case.penalty(w, s, vp, vf, cap=c)
        r = differential_evolution(obj, bounds, vectorized=True, updating="deferred",
                                   integrality=integ, maxiter=100, popsize=16, seed=seed,
                                   tol=1e-8, polish=False, mutation=(0.5, 1.), recombination=0.7)
        X = r.x.reshape(1, -1)
        w, s, vp, vf = case.evaluate(X)
        if case.feasible(w, s, vp, vf)[0] and (best is None or w[0] < best[1]):
            best = (X[0], w[0], s[0], vp[0], vf[0])
    wall = time.time() - t0
    if best is None:
        return None
    X0, w, s, vp, vf = best
    return result_row(case, "de_eps", X0, w, s, vp, vf, wall, case.n_eval)


def run_cma(case: Case, seed=0, sigma0=0.25, restarts=4, maxfevals=40_000):
    """CMA-ES in unit cube with the same penalty (cap=ALLOW), IPOP restarts."""
    import cma
    case.reset(); t0 = time.time(); best = None
    rng = np.random.RandomState(seed)
    budget_per = maxfevals // restarts
    for ri in range(restarts):
        x0 = rng.rand(DIM)
        es = cma.CMAEvolutionStrategy(x0, sigma0, dict(
            bounds=[0, 1], seed=seed + 101 * ri, verbose=-9,
            maxfevals=budget_per, popsize=16 + 4 * ri))
        while not es.stop():
            U = np.array(es.ask())
            X = unit_to_x(np.clip(U, 0, 1))
            w, s, vp, vf = case.evaluate(X)
            f = w + case.penalty(w, s, vp, vf)      # cap defaults to case.cap
            es.tell(list(U), list(f))
            fe = case.feasible(w, s, vp, vf)
            if fe.any():
                i = np.where(fe)[0][np.argmin(w[fe])]
                if best is None or w[i] < best[1]:
                    best = (X[i].copy(), w[i], s[i], vp[i], vf[i])
    wall = time.time() - t0
    if best is None:
        return None
    X0, w, s, vp, vf = best
    return result_row(case, "cma_es", X0, w, s, vp, vf, wall, case.n_eval)
