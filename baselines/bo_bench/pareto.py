"""Mass-stress Pareto front: NSGA-II (published config) vs qLogNEHVI (BO).

Both minimize (mass, stress) s.t. payload/fuel volume constraints. Hypervolume
computed in minimization space against a shared reference point.
"""
import time
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize as pmin

from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from gpytorch.mlls import SumMarginalLogLikelihood

from .problem import Case, DIM, LB, UB, unit_to_x, result_row
from .problem import integ as INTEG

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.double
REF = np.array([300.0, 560.0])          # (mass kg, stress MPa) minimization ref point


def hypervolume(front_ws):
    """front_ws: (n,2) [mass, stress] feasible points; HV vs REF in max-space"""
    if len(front_ws) == 0:
        return 0.0
    Y = torch.tensor(-front_ws, dtype=DT)               # maximize (-mass, -stress)
    ref = torch.tensor(-REF, dtype=DT)
    keep = (Y > ref).all(dim=1)
    if not keep.any():
        return 0.0
    bd = DominatedPartitioning(ref_point=ref, Y=Y[keep])
    return float(bd.compute_hypervolume())


def _front(W, S, mask):
    """nondominated (mass, stress) among mask"""
    P = np.column_stack([W[mask], S[mask]])
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        if keep[i]:
            dom = (P <= P[i]).all(1) & (P < P[i]).any(1)
            if dom.any():
                keep[i] = False
    return P[keep]


def _front_idx(W, S, mask):
    """indices (into the full arrays) of the nondominated (mass, stress) set"""
    ids = np.where(mask)[0]
    P = np.column_stack([W[ids], S[ids]])
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        if keep[i]:
            dom = (P <= P[i]).all(1) & (P < P[i]).any(1)
            if dom.any():
                keep[i] = False
    return ids[keep]


def run_nsga_front(case: Case, seed=1, pop=90, gens=60):
    """published pareto_rad5_fronts.py configuration"""
    case.reset(); t0 = time.time()

    class Pb(Problem):
        def __init__(s):
            super().__init__(n_var=DIM, n_obj=2, n_ieq_constr=2, xl=LB, xu=UB)

        def _evaluate(s, X, out, *a, **k):
            w, st, vp, vf = case.evaluate(X)
            out["F"] = np.column_stack([w, st])
            out["G"] = np.column_stack([(case.VPM - vp) / case.VPM, (case.VFM - vf) / case.VFM])

    r = pmin(Pb(), NSGA2(pop_size=pop, sampling=FloatRandomSampling(),
                         crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20),
                         eliminate_duplicates=True), ("n_gen", gens), seed=seed, verbose=False)
    wall = time.time() - t0
    if r.F is None:
        return None
    order = np.argsort(r.F[:, 0])
    fr = r.F[order]
    Xr = r.X[order].copy(); Xr[:, INTEG] = np.round(Xr[:, INTEG])
    return dict(method="nsga2", case=case.ci, KCAS=case.KCAS, front=fr, X=Xr,
                hv=hypervolume(fr), wall_s=wall, n_eval=case.n_eval)


def run_qlognehvi(case: Case, seed=0, n_init=48, q=4, n_iter=48, device=None):
    """memory-lean: small q, cache_root=False, cache clearing (GPU may be shared)"""
    dev = torch.device(device) if device else DEV
    case.reset(); t0 = time.time()
    torch.manual_seed(seed)
    sobol = SobolEngine(DIM, scramble=True, seed=seed)
    U = sobol.draw(n_init).numpy()
    w, s, vp, vf = case.evaluate(unit_to_x(U))
    Ua, Wa, Sa, VPa, VFa = U, w, s, vp, vf
    bounds01 = torch.stack([torch.zeros(DIM, dtype=DT, device=dev),
                            torch.ones(DIM, dtype=DT, device=dev)])
    ref = torch.tensor(-REF, dtype=DT, device=dev)

    for it in range(n_iter):
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        tX = torch.tensor(Ua, dtype=DT, device=dev)
        # outputs: [-mass, -stress, cons_vp, cons_vf] (cons <= 0 feasible)
        Y = np.column_stack([-Wa, -Sa, (case.VPM - VPa) / case.VPM, (case.VFM - VFa) / case.VFM])
        gps = [SingleTaskGP(tX, torch.tensor(Y[:, j:j + 1], dtype=DT, device=dev),
                            outcome_transform=Standardize(m=1)) for j in range(4)]
        model = ModelListGP(*gps)
        try:
            fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model),
                             optimizer_kwargs={"options": {"maxiter": 50}})
        except Exception:
            pass
        from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model, ref_point=ref, X_baseline=tX, prune_baseline=True,
            objective=WeightedMCMultiOutputObjective(
                weights=torch.ones(2, dtype=DT, device=dev), outcomes=[0, 1]),
            constraints=[lambda Z: Z[..., 2], lambda Z: Z[..., 3]],
            cache_root=False,
        )
        cand, _ = optimize_acqf(acqf, bounds=bounds01, q=q, num_restarts=4, raw_samples=128,
                                options={"maxiter": 50})
        Uq = cand.detach().cpu().numpy()
        w, s, vp, vf = case.evaluate(unit_to_x(Uq))
        Ua = np.vstack([Ua, Uq]); Wa = np.concatenate([Wa, w]); Sa = np.concatenate([Sa, s])
        VPa = np.concatenate([VPa, vp]); VFa = np.concatenate([VFa, vf])

    wall = time.time() - t0
    okv = (VPa >= 0.98 * case.VPM) & (VFa >= 0.98 * case.VFM)
    if not okv.any():
        return None
    idx = _front_idx(Wa, Sa, okv)
    idx = idx[np.argsort(Wa[idx])]
    fr = np.column_stack([Wa[idx], Sa[idx]])
    Xr = unit_to_x(Ua[idx]); Xr[:, INTEG] = np.round(Xr[:, INTEG])
    return dict(method="qlognehvi", case=case.ci, KCAS=case.KCAS, front=fr, X=Xr,
                hv=hypervolume(fr), wall_s=wall, n_eval=case.n_eval)
