import glob
import json
import os
from pathlib import Path

import pandas as pd


def harvest():
    root = (
        Path("D:/SVD_Project")
        if Path("D:/SVD_Project").exists()
        else Path(__file__).resolve().parent.parent
    )
    results_dir = root / "results"
    summary_files = glob.glob(os.path.join(str(results_dir), "**", "*summary*.csv"), recursive=True)
    print(f"Found {len(summary_files)} summary files.")

    all_records = []
    for sf in summary_files:
        try:
            df = pd.read_csv(sf)
            df.columns = [str(c).strip().lower() for c in df.columns]
            for _, row in df.iterrows():
                rec = row.to_dict()
                rec["_source_path"] = sf.replace("\\", "/")
                all_records.append(rec)
        except Exception as e:
            print(f"Error reading {sf}: {e}")

    out_file = root / "harvested_benchmark_data.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, default=str)
    print(f"Successfully harvested {len(all_records)} records to {out_file}")


if __name__ == "__main__":
    harvest()
