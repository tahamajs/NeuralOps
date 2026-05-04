# framework/analysis/report_generator.py
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from .plotting import create_comprehensive_plots, plot_tradeoff

def generate_html_report(df: pd.DataFrame, output_dir: str, title: str = "Analysis Report") -> str:
    os.makedirs(output_dir, exist_ok=True)
    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>body{{font-family:Arial;margin:20px}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ddd;padding:8px}} th{{background:#f2f2f2}}</style>
</head><body>
<h1>{title}</h1><p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<h2>Summary Table</h2>{df.to_html(index=False)}
</body></html>"""
    out = os.path.join(output_dir, 'report.html')
    with open(out, 'w') as f:
        f.write(html)
    return out

def generate_latex_table(df: pd.DataFrame, output_path: str, caption: str = "Results") -> str:
    lines = [r"\begin{table}[htbp]\centering", r"\caption{"+caption+"}", r"\begin{tabular}{lcccr}", r"\toprule"]
    lines.append("Model & Task & Rank & Accuracy & Drop \\\\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        acc = row.get('accuracy', 0)
        drop = row.get('accuracy_drop', '')
        drop_str = f"{drop:.4f}" if isinstance(drop, (int,float)) else '-'
        lines.append(f"{row['model']} & {row['task']} & {row['rank']} & {acc:.4f} & {drop_str} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    return output_path

def generate_full_report(df: pd.DataFrame, output_dir: str, experiment_name: str = "experiment") -> str:
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    create_comprehensive_plots(df, plots_dir)
    plot_tradeoff(df, plots_dir)
    html_path = generate_html_report(df, output_dir, title=f"{experiment_name} Full Report")
    csv_path = os.path.join(output_dir, "summary.csv")
    df.to_csv(csv_path, index=False)
    latex_path = generate_latex_table(df, os.path.join(output_dir, "results.tex"))
    print(f"HTML: {html_path}\nCSV: {csv_path}\nLaTeX: {latex_path}\nPlots: {plots_dir}")
    return html_path