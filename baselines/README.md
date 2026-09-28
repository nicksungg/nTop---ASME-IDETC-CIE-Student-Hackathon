# Optimizer baselines — 8 methods on the BWB structural inverse-design problem

Reference implementations of eight optimizers, run on one shared problem:
**minimise structural mass** subject to rad-5mm hot-spot **stress ≤ 335.33 MPa**
(Al 7075-T6, 503 MPa / 1.5), payload and fuel volume ≥ the seed design's, and mass ≥ 25 kg.
Each optimizer calls the same surrogate models, so the comparison measures the search
strategy, not the model.

| # | Method | Kind | File | Entry point |
|---|---|---|---|---|
| 1 | DE ε-constraint | evolutionary, swept stress cap (the baseline) | [`bo_bench/baselines.py`](bo_bench/baselines.py) | `run_de` |
| 2 | CMA-ES | evolutionary, penalty + IPOP restarts | [`bo_bench/baselines.py`](bo_bench/baselines.py) | `run_cma` |
| 3 | NSGA-II | evolutionary, mass–stress Pareto front | [`bo_bench/pareto.py`](bo_bench/pareto.py) | `run_nsga_front` |
| 4 | SCBO | GP-BO, trust region + constrained Thompson sampling | [`bo_bench/scbo.py`](bo_bench/scbo.py) | `run_scbo` |
| 5 | qLogNEI | GP-BO, constrained LogEI | [`bo_bench/qlognei.py`](bo_bench/qlognei.py) | `run_qlognei` |
| 6 | qLogNEHVI | GP-BO, multi-objective Pareto front | [`bo_bench/pareto.py`](bo_bench/pareto.py) | `run_qlognehvi` |
| 7 | TabPFN-CEI | TabPFN v2 in-context surrogate, EI × P(feasible) | [`bo_bench/tabpfn_bo.py`](bo_bench/tabpfn_bo.py) | `run_pfn_cei` |
| 8 | TabPFN-UCB | TabPFN v2, GIT-BO-style optimistic quantile | [`bo_bench/tabpfn_bo.py`](bo_bench/tabpfn_bo.py) | `run_pfn_ucb` |

Methods 3 and 6 return a mass–stress front. The other six return one lightest feasible design.

---

## Quickstart

Run everything from this `baselines/` folder.

```bash
pip install -r requirements.txt
# or, with conda (exact versions the benchmark was run with):
#   PYTHONNOUSERSITE=1 conda env create -f environment.yml && conda activate bwb-baselines

python bo_bench/smoke_test.py      # all 8 optimizers, tiny budgets, ~30 s on CPU, writes nothing
python bo_bench/run_bench.py       # full benchmark: 6 operating points x 8 methods (~35 min, GPU)
python bo_bench/analyze.py         # tables + results/bo_bench_compare.png
```

Other drivers:

| Script | What it does |
|---|---|
| `bo_bench/seed_study.py` | 3 seeds per single-point method on case 0 (KCAS 33) → `results/bo_bench_seeds.csv` |
| `bo_bench/cap_fronts.py` | turns each single-point method into a front by sweeping 7 stress caps → `results/bo_bench_cap_fronts.pkl` |
| `bo_bench/rerun_fronts.py` | re-runs any missing qLogNEHVI fronts, falling back to CPU on GPU out-of-memory |
| `bo_bench/regen_fronts_x.py` | regenerates the NSGA-II + qLogNEHVI fronts with design vectors, then runs the cap sweep |
| `train_surrogates.py` | retrains the 4 surrogates in `models/` from [`../data/bwb_structures_dataset.csv`](../data/bwb_structures_dataset.csv) |

`run_bench.py`, `seed_study.py` and `cap_fronts.py` write into `results/` and overwrite the shipped files.

To call one optimizer yourself:

```python
from bo_bench.problem import load_cases, Case
from bo_bench.baselines import run_cma

case = Case(0, load_cases().iloc[0])     # operating point 0 (KCAS 33)
r = run_cma(case, seed=0)                # dict: opt_mass, pred_stress, vp, vf, wall_s, n_eval, x, ...
```

---

## The problem

- **Design vector (21 variables):** the 11 structural parameters (3 of them integers) and the
  10 planform parameters from the main dataset. Flight condition (Altitude, KCAS, AOA) is
  fixed per operating point. Bounds, integer handling and feature order come from
  [`models/metadata_v2.json`](models/metadata_v2.json) through [`mission_lite.py`](mission_lite.py).
- **Meshability caps.** Three bounds are tightened so that designs build in nTop:
  Front Spar Chord % ≥ 0.23, Spar Thickness ≥ 1.2 mm, C2/C1 ≤ 0.80
  (see [`bo_bench/problem.py`](bo_bench/problem.py)).
- **Operating points.** [`data/plan_safe_successful.csv`](data/plan_safe_successful.csv) holds 882
  FE-evaluated designs. `load_cases()` keeps the stress-feasible ones, sorts them by KCAS and takes
  6 evenly spaced rows: KCAS 33, 55, 76, 104, 139 and 245. Each seed design's payload and fuel
  volumes become that case's volume constraints.
- **Surrogates** ([`models/`](models/)): four scikit-learn `HistGradientBoostingRegressor`s
  trained on the repo's 13,720-design dataset. Test R²: stress (log10 Pa) 0.761, mass (log10 kg)
  0.997, payload volume 0.998, fuel volume 0.996. `train_surrogates.py` reproduces the shipped
  files exactly.
- **Feasibility:** stress ≤ 1.02 × cap, volumes ≥ 0.98 × target, mass ≥ 25 kg. The 25 kg floor
  keeps optimizers inside the region the dataset covers.

`Case.evaluate` counts evaluations and times them, so every method's cost is reported the same way.

---

## Results (on the surrogates)

Shipped in [`results/`](results/): `bo_bench_results.csv` (6 cases × 6 single-point methods),
`bo_bench_fronts.pkl` (NSGA-II + qLogNEHVI), `bo_bench_seeds.csv`, `bo_bench_cap_fronts.pkl`,
`bo_bench_plans.json` (winning design vectors), and the two figures.

### Lightest feasible mass

Five of the six operating points hit the 25 kg floor for every method, so there the methods tie on
mass and differ only in cost. KCAS 33 (seed design 786 kg) is the case that separates them:

| Method | KCAS 33, 3 seeds (min / median / max, kg) | KCAS 55–245 (kg) |
|---|---|---|
| CMA-ES | **57.4 / 57.5 / 61.4** | 25.0–25.2 |
| qLogNEI | 58.5 / 58.5 / 58.5 | 25.2–28.0 |
| TabPFN-CEI | 58.5 / 58.6 / 65.6 | 25.0 |
| TabPFN-UCB | 60.3 / 61.7 / 86.9 | 25.0–30.5 |
| DE ε-constraint | 64.6 / 64.9 / 66.2 | 25.0 |
| SCBO | 68.2 / 69.0 / 97.8 | 25.0–25.3 |

### Cost (median per case)

| Method | Wall-clock | Surrogate evaluations |
|---|---|---|
| DE ε-constraint | 8.1 s | 169,685 |
| TabPFN-UCB | 8.9 s | 1,008 |
| CMA-ES | 16.5 s | 26,620 |
| TabPFN-CEI | 32.6 s | 1,008 |
| SCBO | 47.1 s | 1,512 |
| qLogNEI | 131 s | 400 |

On a millisecond surrogate, DE's 170k evaluations cost nothing. Against a real FE solve
(hours per evaluation) only the BO methods are affordable, and on this problem they lose
little or no design quality.

### Pareto fronts: NSGA-II vs qLogNEHVI

Hypervolume, reference point 300 kg / 560 MPa, higher is better:

| KCAS | NSGA-II (5,400 evals) | qLogNEHVI (240 evals) |
|---|---|---|
| 33 | 107,424 | **129,438** |
| 55 | 147,101 | **152,847** |
| 76 | **150,675** | 147,205 |
| 104 | **152,288** | 145,248 |
| 139 | **150,459** | 145,527 |
| 245 | **142,967** | 131,797 |

qLogNEHVI wins the two hardest cases with 22× fewer evaluations. NSGA-II gives a much denser
front (about 60 points vs 8) and is about 100× faster in wall-clock on the surrogate.

---

## FE check

The designs each method proposed (single-point winners across the stress-cap sweep, plus the
NSGA-II and qLogNEHVI fronts, 641 in total) were re-run through the full nTop FE pipeline.
[`results/validate_bo_fronts_validated.csv`](results/validate_bo_fronts_validated.csv) has one
row per design: prediction, FE result, `feasible_fe` (FE rad-5mm stress ≤ 335.33 MPa),
`stress_ratio` (FE / predicted) and `mass_ratio`.

| Method | Designs | FE finished | FE-feasible | Median FE/pred stress | Median FE/pred mass |
|---|---|---|---|---|---|
| DE ε-constraint | 42 | 41 | 39 | 0.69 | 1.05 |
| CMA-ES | 42 | 37 | 35 | 1.18 | 0.97 |
| NSGA-II | 343 | 331 | 321 | 1.96 | 0.89 |
| SCBO | 42 | 38 | 34 | 1.06 | 0.98 |
| qLogNEI | 41 | 30 | 25 | 1.20 | 0.84 |
| qLogNEHVI | 48 | 38 | 34 | 1.68 | 0.81 |
| TabPFN-CEI | 41 | 37 | 30 | 1.22 | 0.96 |
| TabPFN-UCB | 42 | 38 | 27 | 1.67 | 0.90 |

Of the 641 designs, 590 finished, 26 failed in nTop (meshing or solver errors) and 25 were still
pending. The stress surrogate is often optimistic, especially for the thin, light designs the
optimizers favour, so check a surrogate optimum in FE before trusting it.

---

## Implementation notes

- **TabPFN-UCB** follows GIT-BO (Yu, Picard & Ahmed, ICLR 2026) without its gradient-informed
  subspace. That subspace targets d ≥ 100 and needs TabPFN input gradients that `tabpfn` 2.1.0
  does not expose. At d = 21 this version searches the full space.
- **TabPFN** 2.1.0's bar-distribution EI/PI only support maximisation, so mass is modelled as
  −log(mass). The model weights download on first use.
- **qLogNEHVI** is written to use little GPU memory (`q=4`, `cache_root=False`, cache clearing
  each iteration) and can run on CPU (`device="cpu"`).
- **Surrogate models** were pickled with scikit-learn 1.6.1. With another version, run
  `python train_surrogates.py` to rebuild them.

References for every method are in [`references.bib`](references.bib).
