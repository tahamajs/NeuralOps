"""
Aggregate multi-seed results and compute CI + p-values.
"""
import glob
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def compare_to_baseline(baseline_metric: float, compressed_values: list) -> dict:
    """One-sample t-test: is the compressed mean != baseline?"""
    arr = np.array(compressed_values)
    if len(arr) < 2 or arr.std(ddof=1) == 0.0:
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "significant_at_0.05": False,
            "effect_size_cohen_d": 0.0,
        }
    t_stat, p_value = stats.ttest_1samp(arr, baseline_metric)
    return {
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant_at_0.05": bool(p_value < 0.05),
        "effect_size_cohen_d": round(float((arr.mean() - baseline_metric) / arr.std(ddof=1)), 4),
    }


def main():
    all_results = []
    for pattern in glob.glob("results/multiseed_*"):
        try:
            for f in glob.glob(f"{pattern}/**/summary.csv", recursive=True):
                df = pd.read_csv(f)
                for _, row in df.iterrows():
                    all_results.append({
                        "module": row.get("module_name", ""),
                        "seed": int(row.get("seed", 42)),
                        "accuracy_pct": float(row.get("accuracy_pct", 0.0) or 0.0),
                        "perplexity": float(row.get("perplexity", 0.0) or 0.0),
                    })
        except Exception as e:
            print(f"Skip {pattern}: {e}")
    
    if not all_results:
        print("No multiseed results found in results/multiseed_*")
        return
        
    groups = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if r["accuracy_pct"] > 0:
            groups[r["module"]]["accuracy_pct"].append(r["accuracy_pct"])
        if r["perplexity"] > 0:
            groups[r["module"]]["perplexity"].append(r["perplexity"])
    
    summary = []
    pvalues = []
    for module, metrics in groups.items():
        for metric_name, values in metrics.items():
            if len(values) < 2:
                continue
            arr = np.array(values)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1))
            ci_95 = stats.t.interval(0.95, len(arr)-1, loc=mean, scale=stats.sem(arr)) if std > 0 else (mean, mean)
            summary.append({
                "module": module,
                "metric": metric_name,
                "n_seeds": len(arr),
                "mean": round(mean, 4),
                "std": round(std, 4),
                "ci_95_low": round(float(ci_95[0]), 4),
                "ci_95_high": round(float(ci_95[1]), 4),
                "min": round(float(arr.min()), 4),
                "max": round(float(arr.max()), 4),
            })
            
            # Baseline comparison (e.g. SST-2 baseline 91.06)
            baseline_val = 91.06 if metric_name == "accuracy_pct" else 41.35
            comp = compare_to_baseline(baseline_val, values)
            comp.update({"module": module, "metric": metric_name, "baseline": baseline_val})
            pvalues.append(comp)
    
    Path("audit").mkdir(parents=True, exist_ok=True)
    if summary:
        df_summary = pd.DataFrame(summary)
        df_summary.to_csv("audit/multiseed_summary_with_ci.csv", index=False)
        print("=== Multi-seed Summary with 95% CI ===")
        print(df_summary.to_string(index=False))
        
    if pvalues:
        df_pvals = pd.DataFrame(pvalues)
        df_pvals.to_csv("audit/pvalues.csv", index=False)
        print("\n=== Baseline Significance (p-values) ===")
        print(df_pvals.to_string(index=False))


if __name__ == "__main__":
    main()
