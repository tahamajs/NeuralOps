import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import numpy as np

def plot_distribution_report(data_dir="./data", output_image="distribution_report.png"):
    csv_files = glob.glob(f"{data_dir}/*.csv")
    if not csv_files:
        print("No CSV files found.")
        return
    names, p0, p1 = [], [], []
    for f in csv_files:
        df = pd.read_csv(f).head(500)
        if "label" in df.columns:
            counts = df["label"].value_counts().sort_index()
            if len(counts) >= 2:
                names.append(Path(f).stem)
                p0.append(counts.get(0,0)/500*100)
                p1.append(counts.get(1,0)/500*100)
    if not names:
        print("No binary datasets found.")
        return
    plt.figure(figsize=(12,6))
    x = np.arange(len(names))
    plt.bar(x-0.2, p0, 0.4, label="Class 0")
    plt.bar(x+0.2, p1, 0.4, label="Class 1")
    plt.xticks(x, names, rotation=45)
    plt.legend()
    plt.title("Class Distribution (Top 500 samples)")
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Saved distribution plot to {output_image}")
