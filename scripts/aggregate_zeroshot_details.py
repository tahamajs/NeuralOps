"""
Aggregate ALL zero-shot details into audit/.
Run after each batch of 10 modules.
"""
import json
import pandas as pd
from pathlib import Path
from collections import Counter

ROOT = Path("D:/SVD_Project")
RESULTS = ROOT / "results" / "zeroshot_parallel"
AUDIT = ROOT / "audit"
AUDIT.mkdir(exist_ok=True)
TASKS = ["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande"]

# 1. Summary
rows = []
for mdir in sorted(RESULTS.iterdir()):
    if not mdir.is_dir():
        continue
    row = {"module": mdir.name}
    for t in TASKS:
        fp = mdir / f"{t}.json"
        if fp.exists():
            d = json.load(open(fp))
            row[f"{t}_acc"] = d.get("accuracy")
            row[f"{t}_n"] = d.get("num_samples")
    accs = [row.get(f"{t}_acc") for t in TASKS if row.get(f"{t}_acc") is not None]
    row["avg"] = round(sum(accs) / len(accs), 4) if accs else None
    row["n_tasks_complete"] = len(accs)
    rows.append(row)

df = pd.DataFrame(rows)
baseline = df[df["module"] == "none"]
if len(baseline):
    b = baseline.iloc[0]["avg"]
    df["retention_pct"] = (df["avg"] / b * 100).round(2)
df.to_csv(AUDIT / "zeroshot_summary.csv", index=False)
print(f"[1] zeroshot_summary.csv: {len(df)} modules")

# 2. Per-class
per_class_rows = []
for mdir in sorted(RESULTS.iterdir()):
    if not mdir.is_dir():
        continue
    for t in TASKS:
        fp = mdir / f"{t}_detailed.json"
        if not fp.exists():
            continue
        d = json.load(open(fp))
        label_counts = Counter(s["label"] for s in d["per_sample"])
        correct_counts = Counter(s["label"] for s in d["per_sample"] if s["correct"])
        for label, cnt in label_counts.items():
            per_class_rows.append({
                "module": mdir.name, "task": t, "class": label,
                "total": cnt, "correct": correct_counts.get(label, 0),
                "accuracy": round(100.0 * correct_counts.get(label, 0) / cnt, 2),
            })
if per_class_rows:
    pd.DataFrame(per_class_rows).to_csv(AUDIT / "zeroshot_per_class.csv", index=False)
    print(f"[2] zeroshot_per_class.csv: {len(per_class_rows)} rows")

# 3. Per-sample parquet
sample_rows = []
for mdir in sorted(RESULTS.iterdir()):
    if not mdir.is_dir():
        continue
    for t in TASKS:
        fp = mdir / f"{t}_detailed.json"
        if not fp.exists():
            continue
        d = json.load(open(fp))
        for s in d["per_sample"]:
            sample_rows.append({
                "module": mdir.name, "task": t, "idx": s["idx"],
                "label": s["label"], "pred": s["pred"], "correct": s["correct"],
                "score_max": max(s["scores"]) if s["scores"] else None,
                "score_min": min(s["scores"]) if s["scores"] else None,
                "score_gap": (max(s["scores"]) - min(s["scores"])) if s["scores"] else None,
            })
if sample_rows:
    df_s = pd.DataFrame(sample_rows)
    df_s.to_parquet(AUDIT / "zeroshot_per_sample.parquet", index=False)
    print(f"[3] zeroshot_per_sample.parquet: {len(sample_rows)} rows")

# 4. Failures
failures = []
for mdir in sorted(RESULTS.iterdir()):
    if not mdir.is_dir():
        continue
    if not (mdir / "run_metadata.json").exists():
        failures.append({"module": mdir.name, "reason": "missing run_metadata.json"})
    for t in TASKS:
        if not (mdir / f"{t}.json").exists():
            failures.append({"module": mdir.name, "reason": f"missing {t}.json"})
if failures:
    pd.DataFrame(failures).to_csv(AUDIT / "zeroshot_failures.csv", index=False)
    print(f"[4] zeroshot_failures.csv: {len(failures)} failures")

print("[DONE]")
