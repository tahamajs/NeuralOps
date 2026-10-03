"""
Aggregate all zero-shot details into audit/ CSVs.
Run after every batch of 10 modules.
"""
import json
import pandas as pd
from pathlib import Path
from collections import Counter

ROOT = Path("D:/SVD_Project") if Path("D:/SVD_Project").exists() else Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "zeroshot_parallel"
AUDIT = ROOT / "audit"
AUDIT.mkdir(exist_ok=True)
TASKS = ["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande"]


def aggregate_summary():
    rows = []
    for mdir in sorted(RESULTS.iterdir()):
        if not mdir.is_dir():
            continue
        row = {"module": mdir.name}
        for t in TASKS:
            fp = mdir / f"{t}.json"
            if fp.exists():
                try:
                    d = json.load(open(fp, encoding="utf-8"))
                    row[f"{t}_acc"] = d.get("accuracy")
                    row[f"{t}_n"] = d.get("num_samples")
                except Exception:
                    pass
        accs = [row.get(f"{t}_acc") for t in TASKS if row.get(f"{t}_acc") is not None]
        row["avg"] = round(sum(accs) / len(accs), 4) if accs else None
        row["n_tasks_complete"] = len(accs)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    # Use none_v2 as baseline if exists, else none
    baseline_row = df[df["module"] == "none_v2"]
    if len(baseline_row) == 0:
        baseline_row = df[df["module"] == "none"]
    if len(baseline_row):
        b = baseline_row.iloc[0]["avg"]
        if b and b > 0:
            df["retention_pct"] = (df["avg"] / b * 100).round(2)
    df.to_csv(AUDIT / "zeroshot_summary.csv", index=False)
    print(f"[1] zeroshot_summary.csv: {len(df)} modules")
    return df


def aggregate_per_class():
    rows = []
    for mdir in sorted(RESULTS.iterdir()):
        if not mdir.is_dir():
            continue
        for t in TASKS:
            fp = mdir / f"{t}_detailed.json"
            if not fp.exists():
                continue
            try:
                d = json.load(open(fp, encoding="utf-8"))
                label_counts = Counter(s["label"] for s in d.get("per_sample", []))
                correct_counts = Counter(s["label"] for s in d.get("per_sample", []) if s.get("correct"))
                for label, cnt in label_counts.items():
                    rows.append({
                        "module": mdir.name, "task": t, "class": label,
                        "total": cnt, "correct": correct_counts.get(label, 0),
                        "accuracy": round(100.0 * correct_counts.get(label, 0) / cnt, 2) if cnt > 0 else 0.0,
                    })
            except Exception:
                pass
    if rows:
        pd.DataFrame(rows).to_csv(AUDIT / "zeroshot_per_class.csv", index=False)
        print(f"[2] zeroshot_per_class.csv: {len(rows)} rows")


def aggregate_per_sample():
    rows = []
    for mdir in sorted(RESULTS.iterdir()):
        if not mdir.is_dir():
            continue
        for t in TASKS:
            fp = mdir / f"{t}_detailed.json"
            if not fp.exists():
                continue
            try:
                d = json.load(open(fp, encoding="utf-8"))
                for s in d.get("per_sample", []):
                    scores = s.get("scores", [])
                    rows.append({
                        "module": mdir.name, "task": t, "idx": s.get("idx"),
                        "label": s.get("label"), "pred": s.get("pred"), "correct": s.get("correct"),
                        "score_max": max(scores) if scores else None,
                        "score_min": min(scores) if scores else None,
                        "score_gap": (max(scores) - min(scores)) if scores else None,
                    })
            except Exception:
                pass
    if rows:
        pd.DataFrame(rows).to_parquet(AUDIT / "zeroshot_per_sample.parquet", index=False)
        print(f"[3] zeroshot_per_sample.parquet: {len(rows)} rows")


def collect_failures():
    rows = []
    for mdir in sorted(RESULTS.iterdir()):
        if not mdir.is_dir():
            continue
        if not (mdir / "run_metadata.json").exists():
            rows.append({"module": mdir.name, "reason": "missing run_metadata.json"})
        for t in TASKS:
            if not (mdir / f"{t}.json").exists():
                rows.append({"module": mdir.name, "reason": f"missing {t}.json"})
    if rows:
        pd.DataFrame(rows).to_csv(AUDIT / "zeroshot_missing.csv", index=False)
        print(f"[4] zeroshot_missing.csv: {len(rows)} missing")


if __name__ == "__main__":
    aggregate_summary()
    aggregate_per_class()
    aggregate_per_sample()
    collect_failures()
    print("[DONE]")
