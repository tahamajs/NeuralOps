#!/usr/bin/env python3
"""
aggregate_rank_ablation.py
Aggregates rank ablation results into pivot tables and Pareto frontiers.

Usage:
    python aggregate_rank_ablation.py --results_dir D:/svd_project/results/rank_ablation
"""
import os
import glob
import argparse
import pandas as pd
import json

RANKS = [32, 64, 128, 192, 256]
DATASETS = ["mnli", "sst2", "agnews"]


def load_ablation_results(results_dir):
    """Load all rank ablation CSVs into a single long-format DataFrame."""
    rows = []
    for dataset in DATASETS:
        for rank in RANKS:
            csv_path = os.path.join(results_dir, f"{dataset}_rank{rank}", "summary.csv")
            if not os.path.exists(csv_path):
                print(f"  [MISSING] {csv_path}")
                continue
            try:
                df = pd.read_csv(csv_path)
                df["dataset"] = dataset
                df["svd_rank"] = rank
                rows.append(df)
                print(f"  [OK] {dataset}/rank{rank}: {len(df)} rows")
            except Exception as e:
                print(f"  [ERROR] {csv_path}: {e}")
    if not rows:
        raise ValueError("No ablation results found!")
    return pd.concat(rows, ignore_index=True)


def make_pivot_table(df, value_col, datasets=None, output_path=None):
    """Create a pivot: rows=module, cols=(dataset, rank), values=value_col."""
    if datasets is None:
        datasets = DATASETS
    sub = df[df["dataset"].isin(datasets)].copy()
    # Filter PASS/PASS_LOW only
    sub = sub[sub["status"].isin(["PASS", "PASS_LOW"])]
    sub["col"] = sub["dataset"] + "_rank" + sub["svd_rank"].astype(str)
    pivot = sub.pivot_table(
        index="module_name", columns="col", values=value_col, aggfunc="first"
    )
    # Sort columns
    col_order = [f"{d}_rank{r}" for d in datasets for r in RANKS if f"{d}_rank{r}" in pivot.columns]
    pivot = pivot[[c for c in col_order if c in pivot.columns]]
    pivot = pivot.round(4)
    if output_path:
        pivot.to_csv(output_path)
        print(f"  Saved: {output_path}")
    return pivot


def find_optimal_rank(df):
    """For each module × dataset, find the rank with best accuracy."""
    results = []
    for (module, dataset), g in df[df["status"].isin(["PASS", "PASS_LOW"])].groupby(["module_name", "dataset"]):
        if g.empty:
            continue
        best = g.loc[g["accuracy_pct"].astype(float).idxmax()]
        results.append({
            "module_name": module,
            "dataset": dataset,
            "best_rank": int(best["svd_rank"]),
            "best_accuracy": float(best["accuracy_pct"]),
            "best_compression_rate": float(best["compression_rate"]) if best["compression_rate"] else None,
        })
    return pd.DataFrame(results).sort_values(["module_name", "dataset"])


def pareto_frontier(df, dataset):
    """Compute Pareto-optimal modules (best accuracy-compression tradeoff) for a dataset at default rank=128."""
    sub = df[(df["dataset"] == dataset) & (df["svd_rank"] == 128) & (df["status"].isin(["PASS", "PASS_LOW"]))].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["accuracy_pct"] = sub["accuracy_pct"].astype(float)
    sub["compression_rate"] = sub["compression_rate"].astype(float)
    # Pareto: a module is dominated if another has BOTH higher accuracy AND higher compression
    pareto = []
    for _, row in sub.iterrows():
        dominated = any(
            (other["accuracy_pct"] >= row["accuracy_pct"] and
             other["compression_rate"] >= row["compression_rate"] and
             (other["accuracy_pct"] > row["accuracy_pct"] or other["compression_rate"] > row["compression_rate"]))
            for _, other in sub.iterrows()
        )
        if not dominated:
            pareto.append(row)
    return pd.DataFrame(pareto).sort_values("compression_rate", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/rank_ablation",
                    help="Directory containing rank ablation result subdirs")
    ap.add_argument("--output_dir", default=None,
                    help="Where to write output CSVs (defaults to results_dir)")
    args = ap.parse_args()

    out_dir = args.output_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=== Loading ablation results ===")
    df = load_ablation_results(args.results_dir)
    print(f"Total rows loaded: {len(df)}")
    df.to_csv(os.path.join(out_dir, "ablation_all_rows.csv"), index=False)

    print("\n=== Table 1: Accuracy vs Rank ===")
    acc_pivot = make_pivot_table(df, "accuracy_pct",
                                  output_path=os.path.join(out_dir, "table1_accuracy_vs_rank.csv"))
    print(acc_pivot.to_string())

    print("\n=== Table 2: Compression Rate vs Rank ===")
    cr_pivot = make_pivot_table(df, "compression_rate",
                                 output_path=os.path.join(out_dir, "table2_compression_vs_rank.csv"))
    print(cr_pivot.to_string())

    print("\n=== Table 3: Tokens/sec vs Rank ===")
    tps_pivot = make_pivot_table(df, "tokens_per_sec",
                                  output_path=os.path.join(out_dir, "table3_tps_vs_rank.csv"))

    print("\n=== Table 4: Optimal Rank per Module per Dataset ===")
    opt = find_optimal_rank(df)
    opt.to_csv(os.path.join(out_dir, "table4_optimal_rank.csv"), index=False)
    print(opt.to_string())

    print("\n=== Table 5: Pareto Frontiers (rank=128) ===")
    all_pareto = []
    for dataset in DATASETS:
        p = pareto_frontier(df, dataset)
        if not p.empty:
            p["dataset"] = dataset
            all_pareto.append(p)
            print(f"\n  {dataset.upper()} Pareto front ({len(p)} modules):")
            print(p[["module_name", "accuracy_pct", "compression_rate"]].to_string(index=False))
    if all_pareto:
        pd.concat(all_pareto).to_csv(os.path.join(out_dir, "table5_pareto_frontier.csv"), index=False)

    print(f"\n✅ All tables written to: {out_dir}")


if __name__ == "__main__":
    main()
