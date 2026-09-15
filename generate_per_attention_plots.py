#!/usr/bin/env python3
"""
Generate publication-quality diagnostic plots directly INSIDE each module's result directory.
Saved locations:
  1. D:\\SVD_Project\\results\\unified_<module>\\<module>_benchmark_profile.png
  2. D:\\SVD_Project\\results\\unified_<module>\\<backbone>\\<module>_accuracy_latency_vram.png
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = r"D:\SVD_Project\results" if os.path.exists(r"D:\SVD_Project\results") else "/Users/tahamajs/Documents/uni/Project/results"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def process_module_dir(mod_dir):
    mod_name = os.path.basename(mod_dir).replace("unified_", "")
    backbones = ["distilbert", "roberta", "gpt2"]
    
    records = []
    
    for b in backbones:
        b_dir = os.path.join(mod_dir, b)
        csv_path = os.path.join(b_dir, "summary.csv")
        
        if not os.path.exists(csv_path):
            continue
            
        try:
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                continue
                
            base_row = df[df['module_name'] == 'full']
            mod_row = df[df['module_name'] != 'full']
            
            b_acc = float(base_row['accuracy_pct'].iloc[0]) if len(base_row) > 0 else 0.0
            m_acc = float(mod_row['accuracy_pct'].iloc[-1]) if len(mod_row) > 0 else 0.0
            status = str(mod_row['status'].iloc[-1]) if len(mod_row) > 0 else "PASS"
            tokens = float(mod_row['tokens_per_sec'].iloc[-1]) if (len(mod_row) > 0 and 'tokens_per_sec' in mod_row) else 0.0
            vram = float(mod_row['peak_vram_mb'].iloc[-1]) if (len(mod_row) > 0 and 'peak_vram_mb' in mod_row) else 0.0
            f1 = float(mod_row['f1_score'].iloc[-1]) if (len(mod_row) > 0 and 'f1_score' in mod_row) else 0.0
            
            records.append({
                "backbone": b.upper(),
                "base_acc": b_acc,
                "mod_acc": m_acc,
                "status": status,
                "tokens": tokens,
                "vram": vram,
                "f1": f1
            })
            
            # 1. Generate per-backbone chart inside the backbone folder
            fig_b, ax_b = plt.subplots(1, 2, figsize=(9, 4))
            ax_b[0].bar(['Baseline', mod_name], [b_acc, m_acc], color=['#BDC3C7', '#2980B9'], width=0.5)
            ax_b[0].set_ylim(0, 105)
            ax_b[0].set_ylabel('Accuracy (%)')
            ax_b[0].set_title(f'{b.upper()}: Validation Accuracy')
            ax_b[0].axhline(56.0, color='#E74C3C', linestyle=':', label='Threshold 56%')
            ax_b[0].legend(loc='lower right', fontsize=8)
            for bar in ax_b[0].patches:
                val = bar.get_height()
                if val > 0:
                    ax_b[0].annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            ax_b[1].bar(['Throughput (tok/s)', 'VRAM (MB)'], [tokens, vram], color=['#27AE60', '#8E44AD'], width=0.5)
            ax_b[1].set_title(f'{b.upper()}: Efficiency Profile')
            ax_b[1].set_yscale('log')
            for bar in ax_b[1].patches:
                val = bar.get_height()
                if val > 0:
                    ax_b[1].annotate(f'{int(val):,}', xy=(bar.get_x() + bar.get_width()/2, val),
                                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            per_backbone_out = os.path.join(b_dir, f"{mod_name}_{b}_profile.png")
            fig_b.savefig(per_backbone_out, dpi=300, bbox_inches='tight')
            plt.close(fig_b)
            
        except Exception:
            pass

    if not records:
        return False
        
    # 2. Generate Master 3-Panel comparison across all backbones inside unified_<module>/
    df_all = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"Mechanism Dissection Profile: {mod_name}", fontsize=13, fontweight='bold', y=1.03)
    
    # Accuracy panel
    x = np.arange(len(df_all))
    width = 0.35
    axes[0].bar(x - width/2, df_all['base_acc'], width, label='Full Baseline', color='#BDC3C7')
    bars = axes[0].bar(x + width/2, df_all['mod_acc'], width, label=mod_name, color='#2980B9')
    axes[0].set_ylabel('Validation Accuracy (%)')
    axes[0].set_title('Accuracy vs Baseline')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_all['backbone'], fontweight='bold')
    axes[0].set_ylim(0, 105)
    axes[0].axhline(56.0, color='#E74C3C', linestyle=':', label='Pass (56%)')
    axes[0].legend(loc='lower right', frameon=True, fontsize=8)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            axes[0].annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
            
    # Throughput panel
    colors = ['#27AE60' if 'PASS' in s else '#E67E22' for s in df_all['status']]
    axes[1].bar(df_all['backbone'], df_all['tokens'], color=colors, width=0.45)
    axes[1].set_ylabel('Tokens / Second')
    axes[1].set_title('Inference Throughput')
    for i, v in enumerate(df_all['tokens']):
        if v > 0:
            axes[1].annotate(f'{int(v):,}', xy=(i, v), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            
    # VRAM panel
    axes[2].bar(df_all['backbone'], df_all['vram'], color='#8E44AD', width=0.45)
    axes[2].set_ylabel('Peak VRAM (MB)')
    axes[2].set_title('Peak GPU Memory')
    for i, v in enumerate(df_all['vram']):
        if v > 0:
            axes[2].annotate(f'{v:.1f} MB', xy=(i, v), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    master_out = os.path.join(mod_dir, f"{mod_name}_dissection_profile.png")
    fig.savefig(master_out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return True

if __name__ == "__main__":
    dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "unified_*")))
    print(f"Scanning {len(dirs)} module directories in {RESULTS_DIR}...")
    success = 0
    for d in dirs:
        if os.path.isdir(d):
            if process_module_dir(d):
                success += 1
                
    print(f"Done! Generated high-res plots directly saved inside {success} module folders.")
