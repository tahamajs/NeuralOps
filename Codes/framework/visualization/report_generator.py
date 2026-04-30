# framework/visualization/report_generator.py
"""
Simple HTML, CSV, and LaTeX report generation.
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional


def generate_csv_summary(all_results: Dict[str, Dict], output_path: str) -> pd.DataFrame:
    """
    Generate CSV summary from results.
    
    Args:
        all_results: Dictionary mapping task to rank to metrics
        output_path: Path to save CSV
        
    Returns:
        DataFrame with summary
    """
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
    print(f"✅ CSV summary saved to {output_path}")
    return df


def generate_html_report(
    results_dir: str,
    output_dir: str,
    df: pd.DataFrame,
    title: str = "SVD-Lin Experiment Report"
) -> str:
    """
    Generate simple HTML report from results.
    
    Args:
        results_dir: Directory containing results
        output_dir: Output directory for report
        df: DataFrame with results
        title: Report title
        
    Returns:
        Path to HTML file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        img {{ max-width: 45%; margin: 10px; border: 1px solid #ccc; border-radius: 5px; }}
        .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>Performance Summary</h2>
        {df.to_html(index=False)}
    </div>
    
    <h2>Visualizations</h2>
    <div>
"""
    
    # Add images
    img_count = 0
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.endswith(".png"):
                rel_path = os.path.relpath(os.path.join(root, f), output_dir)
                html += f'<img src="{rel_path}" alt="{f}">\n'
                img_count += 1
    
    if img_count == 0:
        html += "<p>No visualization images found.</p>"
    
    html += """
    </div>
</body>
</html>"""
    
    output_path = os.path.join(output_dir, "report.html")
    with open(output_path, "w") as f:
        f.write(html)
    
    print(f"✅ HTML report saved to {output_path}")
    return output_path


def generate_latex_table(df: pd.DataFrame, output_path: str, caption: str = "Experimental Results") -> str:
    """
    Generate LaTeX table from results DataFrame.
    
    Args:
        df: DataFrame with results
        output_path: Path to save LaTeX file
        caption: Table caption
        
    Returns:
        Path to LaTeX file
    """
    latex = "\\begin{table}[htbp]\n\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += "\\begin{tabular}{lcccr}\n"
    latex += "\\toprule\n"
    latex += "Model & Task & Rank & Accuracy & Drop \\\\\n"
    latex += "\\midrule\n"
    
    for _, row in df.iterrows():
        acc = row.get('accuracy', row.get('pearson_spearman', 0))
        drop = row.get('accuracy_drop', '')
        drop_str = f"{drop:.4f}" if isinstance(drop, (int, float)) else "-"
        latex += f"{row['model']} & {row['task']} & {row['rank']} & {acc:.4f} & {drop_str} \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"✅ LaTeX table saved to {output_path}")
    return output_path