import glob
import json
import csv
from pathlib import Path

def rebuild():
    pattern = "results/gpt2_sweep/*/result.json"
    files = sorted(glob.glob(pattern))
    rows = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                rows.append(json.load(fp))
        except Exception as e:
            print(f"Error reading {f}: {e}")
    if rows:
        keys = list(rows[0].keys())
        out_csv = Path("results/gpt2_sweep/summary.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"Successfully rebuilt {out_csv} with {len(rows)} modules.")

if __name__ == "__main__":
    rebuild()
