#!/usr/bin/env python3
"""Regenerate all fronts WITH design vectors (same seeds -> same designs):
NSGA-II + qLogNEHVI population fronts -> bo_bench_fronts.pkl
then the cap sweep (cap_fronts.py picks up the empty pkl and refills it)."""
import os, pickle, runpy, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bo_bench.problem import load_cases, Case
from bo_bench.pareto import run_nsga_front, run_qlognehvi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP = os.path.join(ROOT, "results", "bo_bench_fronts.pkl")
P = lambda *a: print(*a, flush=True)

fronts = []
cases = load_cases()
for ci, crow in cases.iterrows():
    case = Case(ci, crow)
    fr = run_nsga_front(case, seed=1)
    fronts.append(fr); P(f"nsga2 case {ci}: HV {fr['hv']:.0f}  {len(fr['front'])} pts (X: {fr['X'].shape})")
    pickle.dump(fronts, open(FP, "wb"))
for ci, crow in cases.iterrows():
    case = Case(ci, crow)
    try:
        fr = run_qlognehvi(case, seed=0)
    except Exception as e:
        P(f"qlognehvi case {ci}: GPU failed ({type(e).__name__}), CPU retry")
        fr = run_qlognehvi(case, seed=0, device="cpu")
    fronts.append(fr); P(f"qlognehvi case {ci}: HV {fr['hv']:.0f}  {len(fr['front'])} pts")
    pickle.dump(fronts, open(FP, "wb"))

P("\n--- population fronts done; starting cap sweep ---\n")
runpy.run_path(os.path.join(ROOT, "bo_bench", "cap_fronts.py"), run_name="__main__")
