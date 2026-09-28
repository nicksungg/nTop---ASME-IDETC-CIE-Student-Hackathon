"""Constrained qLogNEI — 'vanilla BO done right' (LogEI family, NeurIPS 2023;
Hvarfner 2024 dimension-scaled lengthscale priors are the BoTorch>=0.12 default).

ModelListGP: output 0 = -log(mass) (maximized), outputs 1..4 = normalized
constraints (<=0 feasible). Feasibility enters via the acqf `constraints` arg.
"""
import time
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
from gpytorch.mlls import SumMarginalLogLikelihood

from .problem import Case, DIM, unit_to_x, result_row

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.double
BOUNDS01 = torch.stack([torch.zeros(DIM, dtype=DT, device=DEV),
                        torch.ones(DIM, dtype=DT, device=DEV)])


def run_qlognei(case: Case, seed=0, n_init=48, q=8, n_iter=44, method_name="qlognei"):
    case.reset(); t0 = time.time()
    torch.manual_seed(seed)

    sobol = SobolEngine(DIM, scramble=True, seed=seed)
    U = sobol.draw(n_init).numpy()
    w, s, vp, vf = case.evaluate(unit_to_x(U))
    C = case.constraints(w, s, vp, vf)
    Ua, Wa, Sa, VPa, VFa, Ca = U, w, s, vp, vf, C

    for it in range(n_iter):
        tX = torch.tensor(Ua, dtype=DT, device=DEV)
        Y = np.column_stack([-np.log(Wa), Ca])          # (n, 5)
        gps = [SingleTaskGP(tX, torch.tensor(Y[:, j:j + 1], dtype=DT, device=DEV),
                            outcome_transform=Standardize(m=1)) for j in range(Y.shape[1])]
        model = ModelListGP(*gps)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 50}})
        except Exception:
            pass
        acqf = qLogNoisyExpectedImprovement(
            model=model, X_baseline=tX, prune_baseline=True,
            objective=GenericMCObjective(lambda Z, X=None: Z[..., 0]),
            constraints=[(lambda j: (lambda Z: Z[..., j]))(j) for j in range(1, Y.shape[1])],
        )
        cand, _ = optimize_acqf(acqf, bounds=BOUNDS01, q=q, num_restarts=8, raw_samples=256,
                                options={"maxiter": 60})
        Uq = cand.detach().cpu().numpy()
        w, s, vp, vf = case.evaluate(unit_to_x(Uq))
        C = case.constraints(w, s, vp, vf)
        Ua = np.vstack([Ua, Uq]); Wa = np.concatenate([Wa, w]); Sa = np.concatenate([Sa, s])
        VPa = np.concatenate([VPa, vp]); VFa = np.concatenate([VFa, vf]); Ca = np.vstack([Ca, C])

    wall = time.time() - t0
    fe = case.feasible(Wa, Sa, VPa, VFa)
    if not fe.any():
        return None
    idx = np.where(fe)[0][np.argmin(Wa[fe])]
    X0 = unit_to_x(Ua[idx])[0]
    return result_row(case, method_name, X0, Wa[idx], Sa[idx], VPa[idx], VFa[idx],
                      wall, case.n_eval)
