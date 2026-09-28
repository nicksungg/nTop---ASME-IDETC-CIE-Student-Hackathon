#!/usr/bin/env python3
"""Re-run qLogNEHVI fronts (memory-lean) for every case missing one, merging into
results/bo_bench_fronts.pkl."""
import os, pickle, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bo_bench.problem import load_cases, Case
from bo_bench.pareto import run_qlognehvi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP = os.path.join(ROOT, "results", "bo_bench_fronts.pkl")
fronts = pickle.load(open(FP, "rb")) if os.path.exists(FP) else []
have = {(d["case"], d["method"]) for d in fronts}

cases = load_cases()
for ci, crow in cases.iterrows():
    if (ci, "qlognehvi") in have:
        continue
    case = Case(ci, crow)
    try:
        fr = run_qlognehvi(case, seed=0)
    except Exception as e:
        print(f"case {ci}: GPU failed ({type(e).__name__}), retrying on CPU", flush=True)
        fr = run_qlognehvi(case, seed=0, device="cpu")
    if fr:
        fronts.append(fr)
        print(f"case {ci}: HV {fr['hv']:.0f}  {len(fr['front'])} pts  "
              f"{fr['wall_s']:.0f}s  {fr['n_eval']} evals", flush=True)
        pickle.dump(fronts, open(FP, "wb"))
print("DONE", flush=True)
