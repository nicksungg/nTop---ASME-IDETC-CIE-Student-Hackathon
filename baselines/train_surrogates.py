#!/usr/bin/env python
"""Train the 4 rad-5mm surrogates the optimizer benchmark evaluates against:
  stress (worst rad-5mm, log10 Pa) | mass (log10 kg) | payload vol (m3) | fuel vol (m3).

Reads the repo dataset ../data/bwb_structures_dataset.csv ('Max Hotspot Stress' is the
rad-5mm hot-spot stress in MPa). Writes to models/ under the filenames bo_bench/problem.py
loads. The shipped models/*.joblib were produced by this recipe; re-running overwrites them.
"""
import os, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "models")
D = pd.read_csv(os.path.join(HERE, "..", "data", "bwb_structures_dataset.csv"))

STRUCT = ["Skin Thickness", "Front Spar Chord %", "Rear Spar Chord %", "Spar Thickness", "# of Ribs",
          "Rib Thickness", "Wingbox Cutout", "# of Fuselage Ribs", "# of Fuselage Spars",
          "Fuselage Struct Thickness", "Fuselage Struct Width"]
GEOM = ["C2/C1", "C3/C1", "C4/C1", "B1/C1", "B2/C1", "B3/C1", "X3/C1", "S1", "S3", "C1"]
FLIGHT = ["Altitude", "KCAS", "AOA"]
FEATURES = STRUCT + GEOM + FLIGHT          # same order as models/metadata_v2.json
X = D[FEATURES].astype(float).values


def mk():
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=48,
                                         l2_regularization=1.0, min_samples_leaf=20, random_state=0)


def train(name, y, mask, fname):
    m = mask & np.isfinite(y) & np.isfinite(X).all(1)
    Xtr, Xte, ytr, yte = train_test_split(X[m], y[m], test_size=0.2, random_state=0)
    r2 = r2_score(yte, mk().fit(Xtr, ytr).predict(Xte))
    joblib.dump(mk().fit(X[m], y[m]), os.path.join(MD, fname))   # production model: all data
    print(f"  {name:14s}: test R2 = {r2:.3f}   (n={m.sum()})  -> models/{fname}")


print(f"{len(D)} designs | {len(FEATURES)} features\n=== surrogates ===")
s = pd.to_numeric(D["Max Hotspot Stress"], errors="coerce").values        # MPa
w = pd.to_numeric(D["Aircraft Empty Weight"], errors="coerce").values     # kg
allrows = np.ones(len(D), bool)
train("stress(rad5)", np.log10(np.clip(s, 1e-3, None) * 1e6), (s > 0) & (s < 1e4), "stress_rad5_log10_surrogate.joblib")
train("mass",         np.log10(np.clip(w, 1e-6, None)), w > 0, "weight_rad5_log10_surrogate.joblib")
train("payload_vol",  pd.to_numeric(D["Payload Volume"], errors="coerce").values / 1e9, allrows, "payload_vol_rad5_surrogate.joblib")
train("fuel_vol",     pd.to_numeric(D["Fuel Volume"], errors="coerce").values / 1e9, allrows, "fuel_vol_rad5_surrogate.joblib")
print("\ndone - 4 surrogates saved to models/")
