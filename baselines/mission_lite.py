"""Lite mission config for the rad-5mm inverse design.
Exactly the subset the optimizers use (feature layout, bounds, integrality, _assemble),
loaded from models/metadata_v2.json. No aero surrogate, no v2 models — extracted from
inverse_design_mission.py so the bundle is self-contained."""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
meta = json.load(open(os.path.join(MODELS, "metadata_v2.json")))
FEATURES = meta["features"]; STRUCT = meta["struct"]; GEOM = meta["geom"]; FLIGHT = meta["flight"]
ALLOW = meta["allowable_Pa"]; BOUNDS = meta["bounds"]; INT_VARS = meta["int_vars"]
P_VOL  = 5.0e3      # kg per 100% payload-volume shortfall
P_FUEL = 5.0e3      # kg per 100% fuel-volume shortfall
FREE = STRUCT + GEOM                          # 21 design vars (11 structural + 10 planform)
free_pos   = [FEATURES.index(c) for c in FREE]
flight_pos = [FEATURES.index(c) for c in FLIGHT]
bounds     = [tuple(BOUNDS[c]) for c in FREE]
integrality = np.array([c in INT_VARS for c in FREE])
def _assemble(Xf, flight_vals):
    """(N,21) free vars + 3 flight -> (N,len(FEATURES)) surrogate feature matrix."""
    rows = np.empty((Xf.shape[0], len(FEATURES)))
    rows[:, free_pos] = Xf
    rows[:, flight_pos] = flight_vals
    return rows
