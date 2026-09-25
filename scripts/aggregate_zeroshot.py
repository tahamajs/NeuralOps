"""
Aggregate zero-shot reasoning results across evaluated modules.
"""
import json
from pathlib import Path
import pandas as pd


TASKS = ["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande"]


def load_module_results(module_dir: Path) -> dict:
    row = {"module": module_dir.name}
    for t in TASKS:
        fp = module_dir / f"{t}.json"
        if fp.exists():
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    d = json.load(f)
                row[t] = d.get("accuracy", None)
            except Exception:
                row[t] = None
        else:
            row[t] = None
    vals = [row[t] for t in TASKS if row[t] is not None]
    row["avg"] = round(sum(vals) / len(vals), 2) if vals else None
    return row


def main():
    base_dir = Path("results/zeroshot_parallel")
    rows = []
    if base_dir.exists():
        for mdir in base_dir.iterdir():
            if mdir.is_dir():
                rows.append(load_module_results(mdir))

    df = pd.DataFrame(rows)
    if not df.empty:
        baseline_row = df[df["module"] == "none"]
        if len(baseline_row) > 0 and baseline_row.iloc[0]["avg"] is not None:
            baseline_avg = baseline_row.iloc[0]["avg"]
            df["retention_pct"] = (df["avg"] / baseline_avg * 100).round(2)
        else:
            # Fallback to known uncompressed GPT-2 baseline average (41.52)
            df["retention_pct"] = (df["avg"] / 41.52 * 100).round(2)

        out_path = Path("audit/zeroshot_summary.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        sort_col = "retention_pct" if "retention_pct" in df.columns else "avg"
        print(df.sort_values(sort_col, ascending=False).to_string(index=False))
    else:
        print("No zero-shot parallel results found yet.")


if __name__ == "__main__":
    main()
