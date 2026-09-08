import os
import glob
import json
import pandas as pd

def harvest():
    results_dir = "D:/SVD_Project/results"
    summary_files = glob.glob(os.path.join(results_dir, "**", "*summary*.csv"), recursive=True)
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
            
    out_file = "D:/SVD_Project/harvested_benchmark_data.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, default=str)
    print(f"Successfully harvested {len(all_records)} records to {out_file}")

if __name__ == "__main__":
    harvest()
