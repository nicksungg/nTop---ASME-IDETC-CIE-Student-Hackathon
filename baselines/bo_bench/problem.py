"""Shared problem definition for the optimizer benchmark.

Wraps the rad-5mm surrogates into a counted, timed evaluation oracle so every
optimizer (DE / CMA-ES / SCBO / qLogNEI / TabPFN-BO / NSGA-II / qLogNEHVI) sees
the *identical* problem: minimize mass s.t. stress<=335.33, vp>=VPM, vf>=VFM,
mass>=25 (in-distribution floor), with the same meshability-capped bounds and
integer rounding as pareto_rad5.py.
"""
import os, sys, time
import numpy as np, joblib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import mission_lite as M

MD = os.path.join(ROOT, "models")
ALLOW = 335.33
MASS_FLOOR = 25.0

m_s = joblib.load(MD + "/stress_rad5_log10_surrogate.joblib")   # log10(Pa)
m_w = joblib.load(MD + "/weight_rad5_log10_surrogate.joblib")   # log10(kg)
m_v = joblib.load(MD + "/payload_vol_rad5_surrogate.joblib")    # m3
m_f = joblib.load(MD + "/fuel_vol_rad5_surrogate.joblib")       # m3

integ = M.integrality
bounds = list(M.bounds)
# same meshability caps as pareto_rad5.py
bounds[M.FREE.index("Front Spar Chord %")] = (0.23, bounds[M.FREE.index("Front Spar Chord %")][1])
bounds[M.FREE.index("Spar Thickness")]    = (0.0012, bounds[M.FREE.index("Spar Thickness")][1])
bounds[M.FREE.index("C2/C1")]             = (bounds[M.FREE.index("C2/C1")][0], 0.80)
LB = np.array([b[0] for b in bounds]); UB = np.array([b[1] for b in bounds])
DIM = len(bounds)   # 21


def load_cases():
    import pandas as pd
    df = pd.read_csv(os.path.join(ROOT, "data", "plan_safe_successful.csv"))
    df["vp"] = df["Payload Volume"] / 1e9
    df["vf"] = df["Fuel Volume"] / 1e9
    df["w"] = df["Aircraft Empty Weight"]
    df = df[df["Max Stress"] / 1e6 <= ALLOW].sort_values("KCAS").reset_index(drop=True)
    return df.iloc[np.linspace(0, len(df) - 1, 6).astype(int)].reset_index(drop=True)


class Case:
    """One operating point. evaluate(X) counts evals and wall time."""

    def __init__(self, ci, row, cap=ALLOW):
        self.ci = int(ci)
        self.KCAS = float(row.KCAS)
        self.real_mass = float(row.w)
        self.FL = np.array([row.Altitude, row.KCAS, row.AOA])
        self.VPM = float(row.vp); self.VFM = float(row.vf)
        self.cap = float(cap)      # stress cap (MPa); ALLOW unless sweeping a front
        self.reset()

    def reset(self):
        self.n_eval = 0
        self.t_eval = 0.0

    def evaluate(self, X):
        """X (N,21) raw (integers rounded internally) -> mass(kg), stress(MPa), vp, vf."""
        X = np.atleast_2d(np.asarray(X, float)).copy()
        X[:, integ] = np.round(X[:, integ])
        t0 = time.time()
        full = M._assemble(X, self.FL)
        w = 10 ** m_w.predict(full)
        s = 10 ** m_s.predict(full) / 1e6
        vp = m_v.predict(full)
        vf = m_f.predict(full)
        self.t_eval += time.time() - t0
        self.n_eval += len(X)
        return w, s, vp, vf

    # constraint values, <=0 is feasible (normalized like SCBO expects)
    def constraints(self, w, s, vp, vf):
        return np.column_stack([
            (s - self.cap) / self.cap,
            (self.VPM - vp) / self.VPM,
            (self.VFM - vf) / self.VFM,
            (MASS_FLOOR - w) / MASS_FLOOR,
        ])

    def feasible(self, w, s, vp, vf):
        """identical tolerance to pareto_rad5.feas (vs self.cap)"""
        return (s <= self.cap * 1.02) & (vp >= 0.98 * self.VPM) & (vf >= 0.98 * self.VFM) & (w >= MASS_FLOOR)

    def penalty(self, w, s, vp, vf, cap=None):
        if cap is None:
            cap = self.cap
        return (6e4 * np.maximum(0, (s - cap) / cap)
                + M.P_VOL * np.maximum(0, (self.VPM - vp) / self.VPM)
                + M.P_FUEL * np.maximum(0, (self.VFM - vf) / self.VFM)
                + 6e4 * np.maximum(0, (MASS_FLOOR - w) / MASS_FLOOR))


def unit_to_x(U):
    """[0,1]^21 -> raw bounds"""
    return LB + np.atleast_2d(U) * (UB - LB)


def x_to_unit(X):
    return (np.atleast_2d(X) - LB) / (UB - LB)


def result_row(case, method, X_best, w, s, vp, vf, wall, n_eval, extra=None):
    r = dict(case=case.ci, KCAS=case.KCAS, real_mass=case.real_mass, method=method,
             opt_mass=float(w), pred_stress=float(s), vp=float(vp), vf=float(vf),
             pct_lighter=100 * (case.real_mass - float(w)) / case.real_mass,
             wall_s=float(wall), n_eval=int(n_eval),
             feasible=bool(case.feasible(np.array([w]), np.array([s]), np.array([vp]), np.array([vf]))[0]))
    if extra: r.update(extra)
    if X_best is not None:
        r["x"] = list(map(float, X_best))
    return r
