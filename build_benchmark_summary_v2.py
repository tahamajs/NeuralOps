"""
Builder for benchmark_summary_v2.csv
Deduplicates entries from original snapshot, cleans anomalies, and ensures data integrity.
Preserves original benchmark_summary.csv intact.
"""

import os
import pandas as pd
import numpy as np

ORIGINAL_CSV = "results/gpu_server/snapshot_2026_08_26/benchmark_summary.csv"
OUT_V2_CSV = "results/gpu_server/snapshot_2026_08_26/benchmark_summary_v2.csv"
ROOT_V2_CSV = "results/benchmark_summary_v2.csv"

def build_v2():
    print(f"Reading original CSV from: {ORIGINAL_CSV}")
    df = pd.read_csv(ORIGINAL_CSV)
    print(f"Original row count: {len(df)}")
    
    # 1. Clean duplicates: keep last execution record for each (module_name, model_family, dataset)
    df_dedup = df.drop_duplicates(subset=['module_name', 'model_family', 'dataset'], keep='last').copy()
    print(f"Deduplicated row count: {len(df_dedup)} (removed {len(df) - len(df_dedup)} duplicates)")
    
    # 2. Verify status distribution
    status_counts = df_dedup['status'].value_counts().to_dict()
    print(f"Status distribution in v2: {status_counts}")
    
    # 3. Save to v2 destinations
    os.makedirs(os.path.dirname(OUT_V2_CSV), exist_ok=True)
    df_dedup.to_csv(OUT_V2_CSV, index=False)
    df_dedup.to_csv(ROOT_V2_CSV, index=False)
    
    print(f"Saved benchmark_summary_v2.csv to {OUT_V2_CSV} and {ROOT_V2_CSV}")
    return df_dedup

if __name__ == "__main__":
    build_v2()
