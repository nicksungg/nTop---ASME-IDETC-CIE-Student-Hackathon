#!/usr/bin/env python3
"""Analyze bo_bench results: summary tables + comparison figure."""
import os, pickle, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

LABEL = {"de_eps": "DE ε-constraint (baseline)", "cma_es": "CMA-ES",
         "scbo": "SCBO (TR-BO)", "qlognei": "qLogNEI (GP-BO)",
         "pfn_cei": "TabPFN CEI", "pfn_ucb": "TabPFN UCB (GIT-BO-style)"}
COLOR = {"de_eps": "#888888", "cma_es": "#4C72B0", "scbo": "#DD8452",
         "qlognei": "#55A868", "pfn_cei": "#C44E52", "pfn_ucb": "#8172B3"}

df = pd.read_csv(os.path.join(RES, "bo_bench_results.csv"))
df = df[df.feasible == True]

print("=== lightest feasible mass (kg) per case ===")
piv = df.pivot_table(index="KCAS", columns="method", values="opt_mass")
print(piv.round(1).to_string())
print("\n=== wall-clock (s) ===")
print(df.pivot_table(index="KCAS", columns="method", values="wall_s").round(1).to_string())
print("\n=== evaluations ===")
print(df.pivot_table(index="KCAS", columns="method", values="n_eval").round(0).astype(int).to_string())
print("\n=== summary (median across cases) ===")
summ = df.groupby("method").agg(median_mass=("opt_mass", "median"),
                                median_pct=("pct_lighter", "median"),
                                median_wall=("wall_s", "median"),
                                median_evals=("n_eval", "median"),
                                cases_feasible=("case", "nunique"))
# wins: lightest per case
wins = df.loc[df.groupby("case").opt_mass.idxmin(), "method"].value_counts()
summ["wins"] = wins.reindex(summ.index).fillna(0).astype(int)
print(summ.round(2).to_string())

with open(os.path.join(RES, "bo_bench_fronts.pkl"), "rb") as f:
    fronts = pickle.load(f)
fr = pd.DataFrame([{k: v for k, v in d.items() if k != "front"} for d in fronts])
print("\n=== Pareto front hypervolume (ref: 300 kg, 560 MPa) ===")
print(fr.pivot_table(index="KCAS", columns="method", values="hv").round(0).to_string())
print(fr.pivot_table(index="KCAS", columns="method", values="wall_s").round(1).to_string())

# ---- figure ----
fig = plt.figure(figsize=(15, 9.5))
gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.28)

# (a) mass per method per case
ax = fig.add_subplot(gs[0, 0])
methods = [m for m in LABEL if m in df.method.unique()]
cases_k = sorted(df.KCAS.unique())
wdt = 0.8 / len(methods)
for i, m in enumerate(methods):
    sub = df[df.method == m].set_index("KCAS").reindex(cases_k)
    ax.bar(np.arange(len(cases_k)) + i * wdt, sub.opt_mass, wdt,
           label=LABEL[m].split(" (")[0], color=COLOR[m])
ax.set_xticks(np.arange(len(cases_k)) + 0.4)
ax.set_xticklabels([f"{k:.0f}" for k in cases_k])
ax.set_xlabel("KCAS"); ax.set_ylabel("lightest feasible mass (kg)")
ax.set_title("(a) best feasible mass by method"); ax.legend(fontsize=7)

# (b) wall-clock vs mass (median over cases)
ax = fig.add_subplot(gs[0, 1])
for m in methods:
    sub = df[df.method == m]
    ax.scatter(sub.wall_s, sub.opt_mass, color=COLOR[m], s=30, alpha=0.7)
    ax.scatter([sub.wall_s.median()], [sub.opt_mass.median()], color=COLOR[m], s=180,
               marker="*", edgecolor="k", linewidth=0.8, label=LABEL[m].split(" (")[0], zorder=5)
ax.set_xscale("log"); ax.set_xlabel("wall-clock (s, log)"); ax.set_ylabel("best mass (kg)")
ax.set_title("(b) time vs quality (★ = median)"); ax.legend(fontsize=7)

# (c) evals vs mass
ax = fig.add_subplot(gs[0, 2])
for m in methods:
    sub = df[df.method == m]
    ax.scatter(sub.n_eval, sub.opt_mass, color=COLOR[m], s=30, alpha=0.7)
    ax.scatter([sub.n_eval.median()], [sub.opt_mass.median()], color=COLOR[m], s=180,
               marker="*", edgecolor="k", linewidth=0.8, label=LABEL[m].split(" (")[0], zorder=5)
ax.set_xscale("log"); ax.set_xlabel("surrogate evaluations (log)"); ax.set_ylabel("best mass (kg)")
ax.set_title("(c) sample efficiency"); ax.legend(fontsize=7)

# (d-f) Pareto fronts for 3 cases
by_case = {}
for d in fronts:
    by_case.setdefault(d["case"], {})[d["method"]] = d
show = sorted(by_case)[:6][::2] if len(by_case) >= 3 else sorted(by_case)
for j, ci in enumerate(show[:3]):
    ax = fig.add_subplot(gs[1, j])
    for m, col in [("nsga2", "#4C72B0"), ("qlognehvi", "#C44E52")]:
        if m in by_case[ci]:
            d = by_case[ci][m]
            f2 = d["front"]
            ax.step(f2[:, 0], f2[:, 1], where="post", color=col, marker="o", ms=3,
                    label=f"{m}  HV={d['hv']:.0f}  ({d['n_eval']} ev, {d['wall_s']:.0f}s)")
    ax.axhline(335.33, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("mass (kg)"); ax.set_ylabel("rad-5mm stress (MPa)")
    ax.set_title(f"({'def'[j]}) Pareto front, KCAS {by_case[ci]['nsga2']['KCAS']:.0f}")
    ax.legend(fontsize=6.5); ax.set_xlim(0, 300); ax.set_ylim(0, 580)

fig.suptitle("BWB rad-5mm inverse design — optimizer benchmark (BO vs evolutionary)", y=0.98)
out = os.path.join(RES, "bo_bench_compare.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("\nwrote", out)
