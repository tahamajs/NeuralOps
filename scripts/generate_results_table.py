#!/usr/bin/env python3
"""
خواندن نتایج از CSV و تولید جدول LaTeX
"""
import csv
import sys
from pathlib import Path

def generate_table(module_name, csv_path, baseline_path=None):
    if not Path(csv_path).exists():
        return f"% ERROR: CSV not found: {csv_path}"
    
    with open(csv_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    
    baselines = {}
    if baseline_path and Path(baseline_path).exists():
        with open(baseline_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                baselines[row.get('model', '').lower()] = float(row.get('accuracy_pct', 0))
    
    latex = [
        r'\begin{table}[H]',
        r'\centering',
        r'\small',
        f'\\caption{{نتایج بنچمارک استخراج‌شده از داده‌های تجربی برای ماژول \\lr{{{module_name}}}}}',
        f'\\label{{tab:benchmark_{module_name}_csv}}',
        r'\begin{tabular}{lcccccc}',
        r'\toprule',
        r'\textbf{مدل پایه} & \textbf{دقت} & \textbf{F1} & \textbf{حافظه (VRAM)} & \textbf{توان} & \textbf{خط پایه} & \textbf{$\Delta$} \\',
        r'\midrule',
    ]
    
    for row in rows:
        model = row.get('model', row.get('base_model', '--'))
        acc = row.get('accuracy_pct', row.get('accuracy', '--'))
        
        # baseline
        baseline = baselines.get(model.lower())
        if baseline and acc != '--':
            try:
                acc_val = float(acc.replace('%', ''))
                delta = acc_val - baseline
                baseline_str = f'{baseline:.2f}\\%'
                delta_str = f'{delta:+.2f}\\%'
            except:
                baseline_str = '--'
                delta_str = '--'
        else:
            baseline_str = '--'
            delta_str = '--'
        
        f1 = row.get('f1_score', row.get('f1', '--'))
        vram = row.get('peak_vram_mb', row.get('vram_mb', '--'))
        throughput = row.get('throughput_tokens_sec', row.get('throughput', '--'))
        
        try:
            f1_str = f'{float(f1):.4f}' if f1 != '--' else '--'
        except:
            f1_str = str(f1)
            
        try:
            vram_str = f'{float(vram):.1f} MB' if vram != '--' else '--'
        except:
            vram_str = str(vram)
            
        try:
            throughput_str = f'{int(float(throughput)):,} tok/s' if throughput != '--' else '--'
        except:
            throughput_str = str(throughput)
            
        status = row.get('status', 'PASS')
        
        latex.append(f'\\lr{{{model}}} & \\lr{{{acc}\\%}} & \\lr{{{f1_str}}} & \\lr{{{vram_str}}} & \\lr{{{throughput_str}}} & \\lr{{{baseline_str}}} & \\lr{{{delta_str}}} \\\\')
    
    latex.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])
    
    return '\n'.join(latex)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python generate_results_table.py <module_name> <csv_path> [baseline_path]")
        sys.exit(1)
    print(generate_table(sys.argv[1], sys.argv[2], 
                         sys.argv[3] if len(sys.argv) > 3 else None))
