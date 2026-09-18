import pandas as pd
import numpy as np

csv_path = "D:/svd_project/results/a3_optimization/config_sweep.csv"
df = pd.read_csv(csv_path)

print(f"Total configurations recorded so far: {len(df)}")
df["score"] = (df["compression_ratio"] * 0.4) + ((df["accuracy"] / 100.0) * 0.6)

df_sorted = df.sort_values(by="score", ascending=False).reset_index(drop=True)

print("\n=== TOP 5 CONFIGURATIONS ON DISTILBERT ===")
cols = ["config_id", "tau", "rank", "attn_only", "use_fisher", "recal_ln", "compressed_layers", "compression_ratio", "accuracy", "delta_vs_baseline", "score"]
print(df_sorted[cols].head(5).to_string(index=False))

print("\n=== ACCURACY PRESERVATION CHECK (First 3 vs Rest) ===")
print(df[["config_id", "tau", "rank", "accuracy", "compression_ratio"]].head(6).to_string(index=False))
