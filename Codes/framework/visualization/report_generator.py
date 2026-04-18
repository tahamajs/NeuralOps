import os
import json
import pandas as pd
from datetime import datetime

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
    html = f"""<!DOCTYPE html>
<html>
<head><title>SVD-Lin Analysis Report</title>
<style>
body {{ font-family: Arial; margin: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background-color: #f2f2f2; }}
img {{ max-width: 100%; margin: 10px; }}
</style>
</head>
<body>
<h1>SVD-Lin Experiment Report</h1>
<p>Generated: {datetime.now()}</p>
<h2>Performance Summary</h2>
{df.to_html(index=False)}
<h2>Visualizations</h2>
"""
    # Add images (simple listing)
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.endswith(".png"):
                rel_path = os.path.relpath(os.path.join(root, f), output_dir)
                html += f'<img src="{rel_path}" style="max-width:45%">\n'
    html += "</body></html>"
    with open(os.path.join(output_dir, "report.html"), "w") as f:
        f.write(html)
    return html