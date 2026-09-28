"""TabPFN-v2-based Bayesian optimization (two variants).

pfn_cei  — PFN-CEI recipe (Yu, Picard & Ahmed 2024) ported onto TabPFN v2:
           4 in-context regressors (mass, stress, vp, vf); acquisition =
           EI(mass, minimize) x P(stress<=335) x P(vp>=VPM) x P(vf>=VFM) x P(mass>=25),
           argmax over a Sobol + local-perturbation candidate pool (no acq optimizer).

pfn_ucb  — GIT-BO-style (Yu, Picard & Ahmed, ICLR 2026): single TabPFN on the
           penalized objective, optimistic 5%-quantile acquisition on the bar
           distribution, pool argmax. (GIT-BO's gradient-informed subspace targets
           d>=100 and needs input gradients not exposed by tabpfn 2.1.0's public
           API; at d=21 the full space is searched directly, matching the paper's
           observation that subspace restriction only pays off in high dim.)

Both: in-context fitting (no training), integer dims snapped, context capped.
"""
import time
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from tabpfn import TabPFNRegressor

from .problem import (Case, DIM, ALLOW, MASS_FLOOR, unit_to_x, x_to_unit, result_row)
from .problem import integ as INTEG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CTX_MAX = 2500


def _snap(U):
    X = unit_to_x(np.clip(U, 0, 1))
    X[:, INTEG] = np.round(X[:, INTEG])
    return np.clip(x_to_unit(X), 0, 1)


def _pool(rng, sobol, n, center=None, scale=0.15):
    """candidate pool: half global Sobol, half local Gaussian around center"""
    if center is None:
        return _snap(sobol.draw(n).numpy())
    g = sobol.draw(n // 2).numpy()
    loc = np.clip(center + scale * rng.randn(n - n // 2, DIM), 0, 1)
    return _snap(np.vstack([g, loc]))


def _ctx(U, targets, rng, key):
    """cap in-context set: keep the best CTX/2 by key + random rest"""
    n = len(U)
    if n <= CTX_MAX:
        return U, [t for t in targets]
    order = np.argsort(key)
    keep = order[:CTX_MAX // 2]
    rest = rng.choice(order[CTX_MAX // 2:], CTX_MAX - len(keep), replace=False)
    idx = np.concatenate([keep, rest])
    return U[idx], [t[idx] for t in targets]


def _reg(Uc, y):
    r = TabPFNRegressor(device=DEVICE, n_estimators=2)
    r.fit(Uc, y)
    return r


def run_pfn_cei(case: Case, seed=0, n_init=192, q=24, n_iter=34, n_pool=4000,
                method_name="pfn_cei"):
    case.reset(); t0 = time.time()
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    sobol = SobolEngine(DIM, scramble=True, seed=seed)

    U = _snap(sobol.draw(n_init).numpy())
    w, s, vp, vf = case.evaluate(unit_to_x(U))
    Ua, Wa, Sa, VPa, VFa = U, w, s, vp, vf

    for it in range(n_iter):
        fe = case.feasible(Wa, Sa, VPa, VFa)
        best_w = Wa[fe].min() if fe.any() else None
        center = Ua[np.where(fe)[0][np.argmin(Wa[fe])]] if fe.any() else None
        scale = 0.20 * (1 - it / n_iter) + 0.04

        key = Wa + case.penalty(Wa, Sa, VPa, VFa)
        Uc, (Wc, Sc, VPc, VFc) = _ctx(Ua, [Wa, Sa, VPa, VFa], rng, key)
        cand = _pool(rng, sobol, n_pool, center, scale)

        # tabpfn 2.1.0 bar-distribution pi/ei only support maximize=True ->
        # model NEGATED log-mass so min-mass EI and P(w>=floor) are exceed-threshold calls
        r_w = _reg(Uc, -np.log(Wc)); r_s = _reg(Uc, np.log(Sc))
        r_vp = _reg(Uc, VPc);        r_vf = _reg(Uc, VFc)
        o_w = r_w.predict(cand, output_type="full")
        o_s = r_s.predict(cand, output_type="full")
        o_vp = r_vp.predict(cand, output_type="full")
        o_vf = r_vf.predict(cand, output_type="full")
        with torch.no_grad():
            pof = ((1 - o_s["criterion"].pi(o_s["logits"], float(np.log(case.cap))))  # P(s<=cap)
                   * o_vp["criterion"].pi(o_vp["logits"], 0.98 * case.VPM)
                   * o_vf["criterion"].pi(o_vf["logits"], 0.98 * case.VFM)
                   * (1 - o_w["criterion"].pi(o_w["logits"], float(-np.log(MASS_FLOOR)))))  # P(w>=floor)
            if best_w is None:
                acq = pof                      # feasibility search first
            else:
                ei = o_w["criterion"].ei(o_w["logits"], float(-np.log(best_w)))
                acq = ei * pof
            acq = acq.cpu().numpy()
        picks = np.argsort(-acq)[:q]
        Uq = cand[picks]
        w, s, vp, vf = case.evaluate(unit_to_x(Uq))
        Ua = np.vstack([Ua, Uq]); Wa = np.concatenate([Wa, w]); Sa = np.concatenate([Sa, s])
        VPa = np.concatenate([VPa, vp]); VFa = np.concatenate([VFa, vf])

    wall = time.time() - t0
    fe = case.feasible(Wa, Sa, VPa, VFa)
    if not fe.any():
        return None
    idx = np.where(fe)[0][np.argmin(Wa[fe])]
    return result_row(case, method_name, unit_to_x(Ua[idx])[0], Wa[idx], Sa[idx],
                      VPa[idx], VFa[idx], wall, case.n_eval)


def run_pfn_ucb(case: Case, seed=0, n_init=192, q=24, n_iter=34, n_pool=4000,
                quantile=0.05, method_name="pfn_ucb"):
    """GIT-BO-style: single surrogate on penalized objective, optimistic quantile."""
    case.reset(); t0 = time.time()
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    sobol = SobolEngine(DIM, scramble=True, seed=seed)

    U = _snap(sobol.draw(n_init).numpy())
    w, s, vp, vf = case.evaluate(unit_to_x(U))
    Ua, Wa, Sa, VPa, VFa = U, w, s, vp, vf

    for it in range(n_iter):
        F = Wa + case.penalty(Wa, Sa, VPa, VFa)          # penalized objective
        fe = case.feasible(Wa, Sa, VPa, VFa)
        center = Ua[np.where(fe)[0][np.argmin(Wa[fe])]] if fe.any() else Ua[np.argmin(F)]
        scale = 0.20 * (1 - it / n_iter) + 0.04

        Uc, (Fc,) = _ctx(Ua, [F], rng, F)
        cand = _pool(rng, sobol, n_pool, center, scale)
        r_f = _reg(Uc, np.log(Fc))
        o = r_f.predict(cand, output_type="full")
        with torch.no_grad():
            acq = -o["criterion"].icdf(o["logits"], quantile).cpu().numpy()  # optimistic low
        picks = np.argsort(-acq)[:q]
        Uq = cand[picks]
        w, s, vp, vf = case.evaluate(unit_to_x(Uq))
        Ua = np.vstack([Ua, Uq]); Wa = np.concatenate([Wa, w]); Sa = np.concatenate([Sa, s])
        VPa = np.concatenate([VPa, vp]); VFa = np.concatenate([VFa, vf])

    wall = time.time() - t0
    fe = case.feasible(Wa, Sa, VPa, VFa)
    if not fe.any():
        return None
    idx = np.where(fe)[0][np.argmin(Wa[fe])]
    return result_row(case, method_name, unit_to_x(Ua[idx])[0], Wa[idx], Sa[idx],
                      VPa[idx], VFa[idx], wall, case.n_eval)
