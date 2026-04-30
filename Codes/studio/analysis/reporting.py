# studio/analysis/reporting.py
import json
import pandas as pd
from datetime import datetime

def generate_csv_summary(all_results, output_path):
    rows = []
    for task, rank_results in all_results.items():
        for rank, metrics in rank_results.items():
            if "error" in metrics: continue
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
<html><head><title>SVD‑Lin Report</title>
<style>table {{border-collapse: collapse; width: 100%;}} th,td {{border:1px solid #ddd; padding:8px;}}</style>
</head><body><h1>Experiment Report</h1><p>Generated: {datetime.now()}</p>
{df.to_html(index=False)}</body></html>"""
    with open(f"{output_dir}/report.html", "w") as f:
        f.write(html)