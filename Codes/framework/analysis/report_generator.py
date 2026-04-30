# framework/analysis/report_generator.py
"""
HTML, CSV, and LaTeX report generation.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import os


def generate_html_report(
    results_df: pd.DataFrame,
    output_dir: str,
    title: str = "SVD Compression Analysis Report"
) -> str:
    """
    Generate an HTML report from results DataFrame.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save the report
        title: Report title
        
    Returns:
        Path to the generated HTML file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get summary statistics
    summary = results_df.groupby(['model', 'rank']).agg({
        'accuracy': ['mean', 'std', 'count'],
        'rank': 'first'
    }).round(4)
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .metric-good {{ color: green; font-weight: bold; }}
        .metric-bad {{ color: red; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>Experiment Summary</h2>
        <p>Total runs: {len(results_df)}</p>
        <p>Models: {', '.join(results_df['model'].unique())}</p>
        <p>Tasks: {', '.join(results_df['task'].unique())}</p>
    </div>
    
    <h2>Results Summary</h2>
    {summary.to_html()}
    
    <h2>Best Configurations</h2>
    <table>
        <tr><th>Task</th><th>Model</th><th>Rank</th><th>Accuracy</th></tr>
"""
    
    # Add best configurations
    for task in results_df['task'].unique():
        df_task = results_df[results_df['task'] == task]
        if 'accuracy' in df_task.columns:
            best_idx = df_task['accuracy'].idxmax()
            best = df_task.loc[best_idx]
            html += f"""
        <tr>
            <td>{task}</td>
            <td>{best['model']}</td>
            <td>{best['rank']}</td>
            <td class="metric-good">{best['accuracy']:.4f}</td>
        </tr>"""
    
    html += """
    </table>
</body>
</html>"""
    
    output_path = os.path.join(output_dir, 'report.html')
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"✅ HTML report saved to {output_path}")
    return output_path


def generate_csv_summary(results_df: pd.DataFrame, output_path: str) -> str:
    """Generate CSV summary of results."""
    summary = results_df.groupby(['model', 'rank', 'task']).agg({
        'accuracy': 'mean',
        'accuracy_drop': 'mean' if 'accuracy_drop' in results_df.columns else 'first'
    }).round(4)
    
    summary.to_csv(output_path)
    print(f"✅ CSV summary saved to {output_path}")
    return output_path


def generate_latex_table(results_df: pd.DataFrame, output_path: str) -> str:
    """Generate LaTeX table from results."""
    latex = "\\begin{table}[htbp]\n\\centering\n"
    latex += "\\caption{SVD Compression Results}\n"
    latex += "\\begin{tabular}{lcccr}\n"
    latex += "\\toprule\n"
    latex += "Model & Task & Rank & Accuracy & Drop \\\\\n"
    latex += "\\midrule\n"
    
    for _, row in results_df.iterrows():
        latex += f"{row['model']} & {row['task']} & {row['rank']} & {row['accuracy']:.4f} & "
        if 'accuracy_drop' in row:
            latex += f"{row['accuracy_drop']:.4f} \\\\\n"
        else:
            latex += "- \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"✅ LaTeX table saved to {output_path}")
    return output_path