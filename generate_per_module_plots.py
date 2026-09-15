#!/usr/bin/env python3
"""
Publication-Quality Per-Module Visualizer.
Runs directly on the GPU server (or locally), reads all unified_* directories,
and renders a 3-panel publication chart for each module:
1. Validation Accuracy vs Baseline (DistilBERT, RoBERTa, GPT-2)
2. Inference Throughput (tokens/sec)
3. Peak Memory & VRAM Footprint (MB)
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Auto-detect whether running on Windows GPU server or macOS
if os.path.exists(r"D:\SVD_Project\results"):
    RESULTS_DIR = r"D:\SVD_Project\results"
    PLOTS_OUT = r"D:\SVD_Project\results\module_plots"
else:
    RESULTS_DIR = "/Users/tahamajs/Documents/uni/Project/results"
    PLOTS_OUT = "/Users/tahamajs/Documents/uni/Project/thesis_final/module_plots"

os.makedirs(PLOTS_OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def plot_module(mod_dir):
    mod_name = os.path.basename(mod_dir).replace("unified_", "")
    backbones = ["distilbert", "roberta", "gpt2"]
    
    data = []
    for b in backbones:
        csv_path = os.path.join(mod_dir, b, "summary.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if len(df) > 0:
                    base_row = df[df['module_name'] == 'full']
                    mod_row = df[df['module_name'] != 'full']
                    
                    b_acc = float(base_row['accuracy_pct'].iloc[0]) if len(base_row) > 0 else 0.0
                    m_acc = float(mod_row['accuracy_pct'].iloc[-1]) if len(mod_row) > 0 else 0.0
                    m_status = str(mod_row['status'].iloc[-1]) if len(mod_row) > 0 else 'N/A'
                    m_tok = float(mod_row['tokens_per_sec'].iloc[-1]) if (len(mod_row) > 0 and 'tokens_per_sec' in mod_row) else 0.0
                    m_vram = float(mod_row['peak_vram_mb'].iloc[-1]) if (len(mod_row) > 0 and 'peak_vram_mb' in mod_row) else 0.0
                    
                    data.append({
                        "backbone": b.upper(),
                        "base_acc": b_acc,
                        "mod_acc": m_acc,
                        "status": m_status,
                        "tokens_per_sec": m_tok,
                        "peak_vram": m_vram
                    })
            except Exception:
                pass
                
    if not data:
        return False
        
    df_m = pd.DataFrame(data)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.suptitle(f"Mechanism Dissection & Benchmark Profile: {mod_name}", fontsize=13, fontweight='bold', y=1.02)
    
    # 1. Accuracy
    x = np.arange(len(df_m))
    width = 0.35
    axes[0].bar(x - width/2, df_m['base_acc'], width, label='Full Dense Baseline', color='#BDC3C7')
    bars = axes[0].bar(x + width/2, df_m['mod_acc'], width, label=f'{mod_name}', color='#2980B9')
    axes[0].set_ylabel('Validation Accuracy (%)')
    axes[0].set_title('Accuracy Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_m['backbone'], fontweight='bold')
    axes[0].set_ylim(0, 105)
    axes[0].axhline(56.0, color='#E74C3C', linestyle=':', label='Quality Gate (56%)')
    axes[0].legend(loc='lower right', frameon=True, fontsize=8)
    
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            axes[0].annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
            
    # 2. Throughput
    colors = ['#27AE60' if 'PASS' in s else '#E67E22' for s in df_m['status']]
    axes[1].bar(df_m['backbone'], df_m['tokens_per_sec'], color=colors, width=0.45)
    axes[1].set_ylabel('Tokens / Second')
    axes[1].set_title('Inference Throughput')
    for i, v in enumerate(df_m['tokens_per_sec']):
        if v > 0:
            axes[1].annotate(f'{int(v):,}', xy=(i, v), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            
    # 3. Peak VRAM
    axes[2].bar(df_m['backbone'], df_m['peak_vram'], color='#8E44AD', width=0.45)
    axes[2].set_ylabel('Peak VRAM (MB)')
    axes[2].set_title('Memory Allocation')
    for i, v in enumerate(df_m['peak_vram']):
        if v > 0:
            axes[2].annotate(f'{v:.1f} MB', xy=(i, v), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_file = os.path.join(PLOTS_OUT, f"{mod_name}_dissection_profile.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    return True

all_dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "unified_*")))
print(f"Discovered {len(all_dirs)} module directories in {RESULTS_DIR}")

count = 0
for d in all_dirs:
    if os.path.isdir(d):
        if plot_module(d):
            count += 1

print(f"Generated complete diagnostic plots for {count} modules in: {PLOTS_OUT}")
