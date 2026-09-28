"""SCBO — Scalable Constrained Bayesian Optimization (Eriksson & Poloczek, AISTATS 2021).

Trust-region BO with Thompson sampling; constraints handled by the feasibility rule:
prefer feasible min-objective; if none feasible, minimize total violation.
GP per output (objective + each constraint), all modeled in the unit cube.
"""
import time
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from .problem import Case, DIM, unit_to_x, x_to_unit, result_row
from .problem import integ as INTEG

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.double


class TrustRegion:
    def __init__(self, dim, batch_size):
        self.dim = dim
        self.length = 0.8
        self.length_min, self.length_max = 0.5 ** 7, 1.6
        self.failure_tol = max(4, dim // batch_size) if batch_size else dim
        self.success_tol = 3
        self.successes = 0
        self.failures = 0

    def update(self, improved):
        if improved:
            self.successes += 1; self.failures = 0
        else:
            self.failures += 1; self.successes = 0
        if self.successes >= self.success_tol:
            self.length = min(2 * self.length, self.length_max); self.successes = 0
        elif self.failures >= self.failure_tol:
            self.length /= 2; self.failures = 0

    @property
    def dead(self):
        return self.length < self.length_min


def _fit_gp(X, Y, state=None):
    """Near-noiseless GP (deterministic surrogate oracle); warm-start from prior state."""
    from gpytorch.constraints import Interval
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.kernels import MaternKernel, ScaleKernel
    lik = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=X.shape[-1],
                                      lengthscale_constraint=Interval(0.005, 4.0)))
    gp = SingleTaskGP(X, Y.unsqueeze(-1), likelihood=lik, covar_module=kernel,
                      outcome_transform=Standardize(m=1))
    if state is not None:
        try: gp.load_state_dict(state, strict=False)
        except Exception: pass
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    try:
        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 50}})
    except Exception:
        pass
    return gp


def _best_index(W, S_viol):
    """SCBO merit: feasible -> min objective; else min total violation."""
    feas = S_viol <= 0
    if feas.any():
        idx = np.where(feas)[0]
        return idx[np.argmin(W[idx])], True
    return int(np.argmin(S_viol)), False


def _snap_integers(U):
    """snap integer dims of unit-cube candidates to exact integer raw values"""
    X = unit_to_x(U)
    X[:, INTEG] = np.round(X[:, INTEG])
    return np.clip(x_to_unit(X), 0, 1)


def run_scbo(case: Case, seed=0, n_init=48, batch_size=24, max_evals=1500, n_candidates=2048,
             method_name="scbo"):
    case.reset(); t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    states = [None] * 5   # warm-start hyperparameter states: [obj, c1..c4]

    def viol(C):   # total violation per point (C: (N,4) normalized, <=0 feasible)
        return np.clip(C, 0, None).sum(axis=1)

    # global archive (across restarts)
    gX = []; gW = []; gS = []; gVP = []; gVF = []

    def evaluate(U):
        X = unit_to_x(U)
        w, s, vp, vf = case.evaluate(X)
        gX.append(U); gW.append(w); gS.append(s); gVP.append(vp); gVF.append(vf)
        return w, s, vp, vf

    restart = 0
    while case.n_eval < max_evals:
        restart += 1
        tr = TrustRegion(DIM, batch_size)
        sobol = SobolEngine(DIM, scramble=True, seed=seed + 1000 * restart)
        U = sobol.draw(n_init).numpy()
        w, s, vp, vf = evaluate(U)
        C = case.constraints(w, s, vp, vf)
        V = viol(C)
        Us, Ws, Cs, Vs = [U], [w], [C], [V]

        while not tr.dead and case.n_eval < max_evals:
            Ua = np.vstack(Us); Wa = np.concatenate(Ws)
            Ca = np.vstack(Cs); Va = np.concatenate(Vs)
            bi, bfeas = _best_index(Wa, Va)

            tX = torch.tensor(Ua, dtype=DT, device=DEV)
            gp_obj = _fit_gp(tX, torch.log(torch.tensor(Wa, dtype=DT, device=DEV)), states[0])
            states[0] = gp_obj.state_dict()
            gp_con = []
            for j in range(Ca.shape[1]):
                g = _fit_gp(tX, torch.tensor(Ca[:, j], dtype=DT, device=DEV), states[1 + j])
                states[1 + j] = g.state_dict(); gp_con.append(g)

            # trust region box around incumbent, scaled by objective-GP lengthscales
            ls = gp_obj.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
            ls = ls / ls.mean(); ls = ls / np.prod(ls) ** (1 / DIM)
            center = Ua[bi]
            lb = np.clip(center - tr.length / 2 * ls, 0, 1)
            ub = np.clip(center + tr.length / 2 * ls, 0, 1)

            # perturbation candidates (TuRBO-style sparse perturbation)
            cand_sobol = SobolEngine(DIM, scramble=True, seed=rng.randint(1 << 30))
            cand = lb + (ub - lb) * cand_sobol.draw(n_candidates).numpy()
            p = min(1.0, 20 / DIM)
            mask = rng.rand(n_candidates, DIM) <= p
            mask[~mask.any(axis=1), rng.randint(DIM, size=(~mask.any(axis=1)).sum())] = True
            cand = np.where(mask, cand, center)
            cand = _snap_integers(cand)   # proposals are exactly integer-valued

            # Thompson sample all outputs on the candidate set
            tc = torch.tensor(cand, dtype=DT, device=DEV)
            with torch.no_grad():
                f_s = gp_obj.posterior(tc).rsample(torch.Size([batch_size]))[..., 0].cpu().numpy()
                c_s = np.stack([g.posterior(tc).rsample(torch.Size([batch_size]))[..., 0].cpu().numpy()
                                for g in gp_con], axis=-1)   # (q, ncand, 4)
            picks = []
            for q in range(batch_size):
                vq = np.clip(c_s[q], 0, None).sum(axis=1)
                feas_q = vq <= 0
                j = (np.where(feas_q)[0][np.argmin(f_s[q][feas_q])] if feas_q.any()
                     else int(np.argmin(vq)))
                picks.append(j)
            Uq = cand[picks]

            w, s, vp, vf = evaluate(Uq)
            C = case.constraints(w, s, vp, vf)
            V = viol(C)
            Us.append(Uq); Ws.append(w); Cs.append(C); Vs.append(V)

            # success = improved best point under merit rule
            if bfeas:
                improved = ((V <= 0) & (w < Wa[bi] - 1e-3 * abs(Wa[bi]))).any()
            else:
                improved = (V < Va[bi]).any() or (V <= 0).any()
            tr.update(improved)

    wall = time.time() - t0
    Ua = np.vstack(gX); Wa = np.concatenate(gW); Sa = np.concatenate(gS)
    VPa = np.concatenate(gVP); VFa = np.concatenate(gVF)
    fe = case.feasible(Wa, Sa, VPa, VFa)
    if not fe.any():
        return None
    idx = np.where(fe)[0][np.argmin(Wa[fe])]
    X0 = unit_to_x(Ua[idx])[0]
    return result_row(case, method_name, X0, Wa[idx], Sa[idx], VPa[idx], VFa[idx],
                      wall, case.n_eval, extra={"n_restarts": restart})
