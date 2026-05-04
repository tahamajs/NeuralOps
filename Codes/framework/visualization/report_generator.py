"""
Generate CSV, HTML, and LaTeX reports from experiment results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from .utils import set_plot_style, save_figure


def generate_csv_summary(all_results: Dict[str, Dict], output_path: str) -> pd.DataFrame:
    """Create a CSV summary from nested results dict."""
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


def generate_html_report(df: pd.DataFrame, output_dir: str, title: str = "Experiment Report") -> str:
    """Generate a simple HTML report with embedded table."""
    os.makedirs(output_dir, exist_ok=True)
    html = f"""<!DOCTYPE html>
<html>
<head><title>{title}</title>
<style>
    body {{ font-family: Arial; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #f2f2f2; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
{df.to_html(index=False)}
</body>
</html>"""
    out_path = os.path.join(output_dir, "report.html")
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


def generate_latex_table(df: pd.DataFrame, output_path: str, caption: str = "Experimental Results") -> str:
    """Generate a LaTeX table from the DataFrame."""
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\begin{tabular}{lcccr}",
        "\\toprule",
        "Model & Task & Rank & Accuracy & Drop \\\\",
        "\\midrule"
    ]
    for _, row in df.iterrows():
        acc = row.get("accuracy", row.get("pearson_spearman", 0))
        drop = row.get("accuracy_drop", "-")
        drop_str = f"{drop:.4f}" if isinstance(drop, (int, float)) else "-"
        lines.append(f"{row['model']} & {row['task']} & {row['rank']} & {acc:.4f} & {drop_str} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


def plot_accuracy_vs_rank(df: pd.DataFrame, task: Optional[str] = None,
                          hue: str = "model", save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Line plot of accuracy vs rank per model."""
    set_plot_style()
    if task:
        df = df[df["task"] == task]
    fig, ax = plt.subplots(figsize=figsize)
    for group in df[hue].unique():
        data = df[df[hue] == group].sort_values("rank")
        ax.plot(data["rank"], data["accuracy"], marker="o", label=group, linewidth=2, markersize=8)
    ax.set_xlabel("SVD Rank")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs Rank" + (f" - {task.upper()}" if task else ""))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_radar_chart(df: pd.DataFrame, models: List[str], tasks: List[str],
                     metric: str = "accuracy", save_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    """Radar chart comparing multiple models across tasks."""
    set_plot_style()
    angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    for model, color in zip(models, colors):
        values = []
        for task in tasks:
            val = df[(df["model"] == model) & (df["task"] == task)][metric].mean()
            values.append(val if not np.isnan(val) else 0)
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=model)
        ax.fill(angles, values, alpha=0.1, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Radar Chart")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig