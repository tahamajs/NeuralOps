"""Model comparison utilities for SVD compression analysis."""
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional
from .plotting import create_comprehensive_plots, plot_tradeoff

class ModelComparison:
    def __init__(self, results_dirs: List[str], output_dir: str = "./results/comparison"):
        self.results_dirs = results_dirs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataframes = []
        self.df_all = None

    def load_all_results(self) -> pd.DataFrame:
        for d in self.results_dirs:
            csv_path = os.path.join(d, "performance.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df["run"] = os.path.basename(d)
                self.dataframes.append(df)
        if self.dataframes:
            self.df_all = pd.concat(self.dataframes, ignore_index=True)
        return self.df_all

    def generate_all_plots(self):
        if self.df_all is None:
            self.load_all_results()
        plots_dir = self.output_dir / "plots"
        create_comprehensive_plots(self.df_all, str(plots_dir))
        plot_tradeoff(self.df_all, str(plots_dir))
        print(f"Plots saved to {plots_dir}")

def quick_compare(results_dirs: List[str], output_dir: str = "./results/comparison"):
    comparator = ModelComparison(results_dirs, output_dir)
    comparator.load_all_results()
    comparator.generate_all_plots()