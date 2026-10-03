"""HTML, LaTeX, interactive report generators."""

import json
import os
from datetime import datetime

import pandas as pd


def generate_html_report(df: pd.DataFrame, output_dir: str, title: str = "Analysis Report") -> str:
    """Generate html report."""
    os.makedirs(output_dir, exist_ok=True)
    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>body{{font-family:Arial;margin:20px}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ddd;padding:8px}} th{{background:#f2f2f2}}</style>
</head><body>
<h1>{title}</h1><p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
<h2>Summary Table</h2>{df.to_html(index=False)}
</body></html>"""
    out = os.path.join(output_dir, "report.html")
    with open(out, "w") as f:
        f.write(html)
    return out


def generate_latex_table(df: pd.DataFrame, output_path: str, caption: str = "Results") -> str:
    """Generate latex table."""
    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{" + caption + "}",
        r"\begin{tabular}{lcccr}",
        r"\toprule",
    ]
    lines.append("Model & Task & Rank & Accuracy & Drop \\\\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        acc = row.get("accuracy", 0)
        drop = row.get("accuracy_drop", "-")
        drop_str = f"{drop:.4f}" if isinstance(drop, (int, float)) else "-"
        lines.append(
            f"{row['model']} & {row['task']} & {row['rank']} & {acc:.4f} & {drop_str} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


def generate_full_report(
    df: pd.DataFrame, output_dir: str, experiment_name: str = "experiment"
) -> str:
    """Generate full report."""
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    from .plotting import create_comprehensive_plots, plot_tradeoff

    create_comprehensive_plots(df, plots_dir)
    plot_tradeoff(df, plots_dir)
    generate_html_report(df, output_dir, title=f"{experiment_name} Full Report")
    csv_path = os.path.join(output_dir, "summary.csv")
    df.to_csv(csv_path, index=False)
    generate_latex_table(df, os.path.join(output_dir, "results.tex"))
    return os.path.join(output_dir, "report.html")


def generate_interactive_report(df: pd.DataFrame, output_dir: str) -> str:
    """Generate interactive report."""
    os.makedirs(output_dir, exist_ok=True)
    import plotly.graph_objects as go

    fig1 = go.Figure()
    for task in df["task"].unique():
        sub = df[df["task"] == task]
        fig1.add_trace(
            go.Scatter(x=sub["rank"], y=sub["accuracy"], mode="lines+markers", name=task)
        )
    fig1.update_layout(title="Accuracy vs Rank", xaxis_title="Rank", yaxis_title="Accuracy")
    fig2 = go.Figure()
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        fig2.add_trace(
            go.Scatter(x=sub["rank"], y=sub["accuracy"], mode="lines+markers", name=model)
        )
    fig2.update_layout(title="Model Comparison", xaxis_title="Rank", yaxis_title="Accuracy")
    html = f"""<!DOCTYPE html>
<html><head><script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head>
<body><h1>Interactive Report</h1>
<div id="rankPlot"></div><div id="modelPlot"></div>
<script>
Plotly.newPlot('rankPlot', {json.dumps(fig1.to_dict())});
Plotly.newPlot('modelPlot', {json.dumps(fig2.to_dict())});
</script></body></html>"""
    path = os.path.join(output_dir, "interactive_report.html")
    with open(path, "w") as f:
        f.write(html)
    return path
