#!/usr/bin/env python3
"""
Generate an HTML report from experiment results.
Usage:
    python scripts/generate_report.py --results_dir results/run_20250418_143022 --output_dir report
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path to import framework modules if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from framework.visualization.report_generator import generate_csv_summary, generate_html_report
except ImportError:
    # Fallback: define basic functions here
    def generate_csv_summary(all_results, output_path):
        rows = []
        for task, rank_results in all_results.items():
            for rank, metrics in rank_results.items():
                if "error" in metrics:
                    continue
                row = {"task": task, "rank": rank}
                for k, v in metrics.items():
                    if k.startswith("eval_"):
                        row[k.replace("eval_", "")] = v
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        return df

    def generate_html_report(results_dir, output_dir, df):
        # Create a simple HTML report
        os.makedirs(output_dir, exist_ok=True)
        html = f"""<!DOCTYPE html>
<html>
<head><title>SVD-Lin Analysis Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background-color: #f2f2f2; }}
img {{ max-width: 100%; margin: 10px 0; }}
.summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
</style>
</head>
<body>
<h1>SVD-Lin Experiment Report</h1>
<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<div class="summary">
<h2>Performance Summary</h2>
{df.to_html(index=False)}
</div>
<h2>Visualizations</h2>
<div class="vis">
"""
        # Include all PNG images from the results directory
        img_count = 0
        for root, dirs, files in os.walk(results_dir):
            for f in files:
                if f.endswith(".png"):
                    rel_path = os.path.relpath(os.path.join(root, f), output_dir)
                    html += f'<div style="display:inline-block; margin:10px;"><img src="{rel_path}" style="max-width:400px; border:1px solid #ddd; padding:5px;"></div>\n'
                    img_count += 1
        if img_count == 0:
            html += "<p>No visualization images found.</p>"
        html += """
</div>
</body>
</html>"""
        with open(os.path.join(output_dir, "report.html"), "w") as f:
            f.write(html)
        return html


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HTML report from experiment results")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the run folder (e.g., results/run_20250418_143022)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for the report (default: same as results_dir)")
    parser.add_argument("--no_plots", action="store_true", help="Skip generating new plots, only use existing images")
    return parser.parse_args()


def load_all_results(results_dir):
    """Load all summary.json files from the run folder."""
    all_results = {}
    for task_dir in Path(results_dir).iterdir():
        if not task_dir.is_dir():
            continue
        summary_file = task_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                all_results[task_dir.name] = json.load(f)
    return all_results


def generate_rank_plots(all_results, output_dir):
    """Generate accuracy vs rank plots for each task."""
    os.makedirs(output_dir, exist_ok=True)
    for task, task_results in all_results.items():
        ranks = []
        scores = []
        for rank_str, metrics in task_results.items():
            if rank_str == "None" or "error" in metrics:
                continue
            rank = int(rank_str)
            # Extract the primary metric
            if task == "stsb":
                score = metrics.get("pearson", metrics.get("spearman", 0.0))
            else:
                score = metrics.get("accuracy", metrics.get("matthews_correlation", 0.0))
            ranks.append(rank)
            scores.append(score)
        if not ranks:
            continue
        # Sort by rank
        sorted_pairs = sorted(zip(ranks, scores))
        ranks, scores = zip(*sorted_pairs)
        plt.figure(figsize=(8, 5))
        plt.plot(ranks, scores, marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.xlabel("Rank")
        plt.ylabel("Score")
        plt.title(f"{task.upper()} - Performance vs Rank")
        plt.grid(True, alpha=0.3)
        # Add full attention baseline if available
        if "None" in task_results and "error" not in task_results["None"]:
            full_score = task_results["None"].get("accuracy", task_results["None"].get("matthews_correlation", task_results["None"].get("pearson", 0.0)))
            plt.axhline(y=full_score, color='red', linestyle='--', label=f'Full Attention: {full_score:.3f}')
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"rank_comparison_{task}.png"), dpi=150)
        plt.close()
    print(f"Generated rank comparison plots in {output_dir}")


def main():
    args = parse_args()
    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    all_results = load_all_results(results_dir)
    if not all_results:
        print(f"No summary.json files found in {results_dir}")
        sys.exit(1)

    # Generate CSV summary
    csv_path = os.path.join(output_dir, "performance.csv")
    df = generate_csv_summary(all_results, csv_path)
    print(f"CSV summary saved to {csv_path}")

    # Generate rank plots if not disabled
    if not args.no_plots:
        generate_rank_plots(all_results, output_dir)

    # Generate HTML report
    html_path = generate_html_report(results_dir, output_dir, df)
    print(f"HTML report saved to {html_path}")

    # Also copy any existing images from subdirectories if they are not already in output_dir
    import shutil
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.endswith(".png") and "visualizations" in root:
                src = os.path.join(root, f)
                dst = os.path.join(output_dir, f)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"Copied {f} to {output_dir}")

    print("\n✅ Report generation complete.")
    print(f"Open {os.path.join(output_dir, 'report.html')} in a browser.")


if __name__ == "__main__":
    main()