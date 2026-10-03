import os
import sys
import csv
from pathlib import Path
import pandas as pd
import numpy as np

CSV_PATH = Path("results/rank_sweep/sweep_results.csv")
PER_MODULE_DIR = Path("results/rank_sweep/per_module")
PER_MODULE_DIR.mkdir(parents=True, exist_ok=True)

if not CSV_PATH.exists():
    print(f"Error: {CSV_PATH} not found.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)

# Clean numeric columns
for col in ["accuracy_pct", "param_ratio", "num_compressed_layers", "elapsed_s", "requested_rank"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Successful / passing runs
ok = df[df["status"].isin(["PASS", "PASS_LOW", "PASS_NO_COMPRESSION"])].copy()

print("=" * 100)
print("RANK SWEEP ANALYSIS")
print("=" * 100)
print()

# ---- 1. Per-module best rank ----
print("1. Best rank per module (highest accuracy with param_ratio < 0.9):")
print()

best_rows = []
for mod in sorted(df["module"].dropna().unique()):
    sub = ok[(ok["module"] == mod) & (ok["param_ratio"] < 0.9)]
    if sub.empty:
        # Check if any ok run exists without ratio restriction
        sub_all = ok[ok["module"] == mod]
        if sub_all.empty:
            best_rows.append(
                {
                    "module": mod,
                    "best_rank": "—",
                    "accuracy": "—",
                    "ratio": "—",
                    "status": "FAIL/NONE",
                }
            )
        else:
            best = sub_all.sort_values("accuracy_pct", ascending=False).iloc[0]
            best_rows.append(
                {
                    "module": mod,
                    "best_rank": int(best["requested_rank"])
                    if pd.notnull(best["requested_rank"])
                    else "—",
                    "accuracy": round(best["accuracy_pct"], 2)
                    if pd.notnull(best["accuracy_pct"])
                    else "—",
                    "ratio": round(best["param_ratio"], 3)
                    if pd.notnull(best["param_ratio"])
                    else "—",
                    "status": f"{best['status']} (ratio >= 0.9)",
                }
            )
        continue
    best = sub.sort_values("accuracy_pct", ascending=False).iloc[0]
    best_rows.append(
        {
            "module": mod,
            "best_rank": int(best["requested_rank"]) if pd.notnull(best["requested_rank"]) else "—",
            "accuracy": round(best["accuracy_pct"], 2),
            "ratio": round(best["param_ratio"], 3),
            "status": best["status"],
        }
    )

print(pd.DataFrame(best_rows).to_string(index=False))
print()

# ---- 2. Per-rank behavior ----
print("2. Mean behavior by requested rank (all modules):")
print()
if not ok.empty:
    rank_summary = (
        ok.groupby("requested_rank")
        .agg(
            {
                "accuracy_pct": "mean",
                "param_ratio": "mean",
                "num_compressed_layers": "mean",
                "elapsed_s": "mean",
            }
        )
        .round(3)
    )
    print(rank_summary.to_string())
else:
    print("No passing runs to summarize.")
print()

# ---- 3. Per model x rank ----
print("3. Param ratio matrix: model x rank:")
print()
if not ok.empty:
    pivot = ok.pivot_table(
        index="requested_rank", columns="model", values="param_ratio", aggfunc="mean"
    ).round(3)
    print(pivot.to_string())
    print()
    print("Accuracy matrix: model x rank:")
    pivot_acc = ok.pivot_table(
        index="requested_rank", columns="model", values="accuracy_pct", aggfunc="mean"
    ).round(2)
    print(pivot_acc.to_string())
print()

# ---- 4. Modules that never compress ----
print("4. Modules that never achieve param_ratio < 1.0:")
print()
if not df.empty:
    never = df.groupby("module").agg({"param_ratio": "min"}).round(3)
    never_comp = never[never["param_ratio"] >= 1.0]
    if not never_comp.empty:
        print(never_comp.to_string())
    else:
        print("None — all tested modules achieved param_ratio < 1.0 at least once.")
print()

# ---- 5. Save per-module CSV ----
for mod in df["module"].dropna().unique():
    sub = df[df["module"] == mod]
    sub.to_csv(PER_MODULE_DIR / f"{mod}.csv", index=False)
print(f"Per-module CSVs saved to {PER_MODULE_DIR}")
