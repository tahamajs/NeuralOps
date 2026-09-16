#!/usr/bin/env python3
"""
Generate complete visualization suite directly INSIDE each module's results folder:
1. <module>_dissection_profile.png (Accuracy, Throughput, VRAM across 3 backbones)
2. <backbone>_loss_curve.png (Step-by-step recovery loss convergence)
3. <backbone>_radar_metrics.png (Multivariate accuracy, F1, precision, recall radar)
4. <backbone>_layer_ranks.png (SVD / LoRA rank distribution per transformer layer)
"""

import os
import re
import glob
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = r"D:\SVD_Project\results"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def parse_loss(log_path):
    steps, losses = [], []
    if not os.path.exists(log_path):
        return steps, losses
    pat = re.compile(r"fine-tune step (\d+)/\d+ loss=([0-9\.]+)")
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = pat.search(line)
                if m:
                    steps.append(int(m.group(1)))
                    losses.append(float(m.group(2)))
    except Exception:
        pass
    return steps, losses

def plot_loss(steps, losses, out_path, title):
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, losses, marker='o', markersize=3, color='#E74C3C', linewidth=1.8, label='Recovery Loss')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Recovery Step')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(frameon=True, fontsize=8)
    plt.tight_layout()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')
    import gc
    gc.collect()

def plot_radar(metrics, out_path, title):
    categories = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    values = [
        metrics.get('accuracy_pct', 0.0) / 100.0,
        metrics.get('f1_score', 0.0),
        metrics.get('precision', 0.0),
        metrics.get('recall', 0.0)
    ]
    if all(v == 0.0 for v in values):
        return
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]
    
    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='#2980B9', linewidth=2)
    ax.fill(angles, values, color='#3498DB', alpha=0.35)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=11, fontweight='bold', y=1.08)
    plt.tight_layout()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')
    import gc
    gc.collect()

def plot_ranks(ranks_dict, out_path, title):
    if not ranks_dict:
        return
    ranks = list(ranks_dict.values())
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bars = ax.bar(range(1, len(ranks) + 1), ranks, color='#8E44AD', width=0.55)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Transformer Layer Index')
    ax.set_ylabel('Rank')
    ax.set_xticks(range(1, len(ranks) + 1))
    ax.set_ylim(0, max(ranks) * 1.25 if max(ranks) > 0 else 64)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    plt.tight_layout()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')
    import gc
    gc.collect()

# Scan all unified directories
dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "unified_*")))
print(f"Processing {len(dirs)} directories...")

processed = 0
for d in dirs:
    if not os.path.isdir(d):
        continue
    mod_name = os.path.basename(d).replace("unified_", "")
    
    for b in ["distilbert", "roberta", "gpt2"]:
        b_dir = os.path.join(d, b)
        if not os.path.exists(b_dir):
            continue
            
        # 1. Loss curve
        console_log = os.path.join(b_dir, "console.log")
        steps, losses = parse_loss(console_log)
        if steps:
            plot_loss(steps, losses, os.path.join(b_dir, f"{b}_loss_curve.png"), f"{mod_name} ({b.upper()}): Loss Convergence")
            plot_loss(steps, losses, os.path.join(d, f"{mod_name}_{b}_loss_curve.png"), f"{mod_name} ({b.upper()}): Loss Convergence")
            
        # 2. Radar & Layer ranks
        for mf in glob.glob(os.path.join(b_dir, "*_full_metrics.json")):
            if "_full_full_metrics" in mf:
                continue
            try:
                with open(mf, 'r', encoding='utf-8') as f:
                    mdata = json.load(f)
                perf = mdata.get('performance', {})
                eff = mdata.get('efficiency', {})
                
                plot_radar(perf, os.path.join(b_dir, f"{b}_radar_metrics.png"), f"{mod_name} ({b.upper()}): Performance Radar")
                plot_radar(perf, os.path.join(d, f"{mod_name}_{b}_radar_metrics.png"), f"{mod_name} ({b.upper()}): Performance Radar")
                
                ranks = eff.get('per_layer_ranks', {})
                if ranks:
                    plot_ranks(ranks, os.path.join(b_dir, f"{b}_layer_ranks.png"), f"{mod_name} ({b.upper()}): Layer Rank Distribution")
                    plot_ranks(ranks, os.path.join(d, f"{mod_name}_{b}_layer_ranks.png"), f"{mod_name} ({b.upper()}): Layer Rank Distribution")
            except Exception:
                pass
    processed += 1

print(f"Successfully generated all diagnostic plots across {processed} module directories!")

#!/usr/bin/env python3
"""
Generate complete visualization suite directly INSIDE each module's results folder:
1. <module>_dissection_profile.png (Accuracy, Throughput, VRAM across 3 backbones)
2. <backbone>_loss_curve.png (Step-by-step recovery loss convergence)
3. <backbone>_radar_metrics.png (Multivariate accuracy, F1, precision, recall radar)
4. <backbone>_layer_ranks.png (SVD / LoRA rank distribution per transformer layer)
"""

import os
import re
import glob
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = r"D:\SVD_Project\results"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def parse_loss(log_path):
    steps, losses = [], []
    if not os.path.exists(log_path):
        return steps, losses
    pat = re.compile(r"fine-tune step (\d+)/\d+ loss=([0-9\.]+)")
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = pat.search(line)
                if m:
                    steps.append(int(m.group(1)))
                    losses.append(float(m.group(2)))
    except Exception:
        pass
    return steps, losses

def plot_loss(steps, losses, out_path, title):
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, losses, marker='o', markersize=3, color='#E74C3C', linewidth=1.8, label='Recovery Loss')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Recovery Step')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(frameon=True, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_radar(metrics, out_path, title):
    categories = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    values = [
        metrics.get('accuracy_pct', 0.0) / 100.0,
        metrics.get('f1_score', 0.0),
        metrics.get('precision', 0.0),
        metrics.get('recall', 0.0)
    ]
    if all(v == 0.0 for v in values):
        return
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]
    
    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='#2980B9', linewidth=2)
    ax.fill(angles, values, color='#3498DB', alpha=0.35)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=11, fontweight='bold', y=1.08)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_ranks(ranks_dict, out_path, title):
    if not ranks_dict:
        return
    ranks = list(ranks_dict.values())
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bars = ax.bar(range(1, len(ranks) + 1), ranks, color='#8E44AD', width=0.55)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Transformer Layer Index')
    ax.set_ylabel('Rank')
    ax.set_xticks(range(1, len(ranks) + 1))
    ax.set_ylim(0, max(ranks) * 1.25 if max(ranks) > 0 else 64)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Scan all unified directories
dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "unified_*")))
print(f"Processing {len(dirs)} directories...")

processed = 0
for d in dirs:
    if not os.path.isdir(d):
        continue
    mod_name = os.path.basename(d).replace("unified_", "")
    
    for b in ["distilbert", "roberta", "gpt2"]:
        b_dir = os.path.join(d, b)
        if not os.path.exists(b_dir):
            continue
            
        # 1. Loss curve
        console_log = os.path.join(b_dir, "console.log")
        steps, losses = parse_loss(console_log)
        if steps:
            plot_loss(steps, losses, os.path.join(b_dir, f"{b}_loss_curve.png"), f"{mod_name} ({b.upper()}): Loss Convergence")
            plot_loss(steps, losses, os.path.join(d, f"{mod_name}_{b}_loss_curve.png"), f"{mod_name} ({b.upper()}): Loss Convergence")
            
        # 2. Radar & Layer ranks
        for mf in glob.glob(os.path.join(b_dir, "*_full_metrics.json")):
            if "_full_full_metrics" in mf:
                continue
            try:
                with open(mf, 'r', encoding='utf-8') as f:
                    mdata = json.load(f)
                perf = mdata.get('performance', {})
                eff = mdata.get('efficiency', {})
                
                plot_radar(perf, os.path.join(b_dir, f"{b}_radar_metrics.png"), f"{mod_name} ({b.upper()}): Performance Radar")
                plot_radar(perf, os.path.join(d, f"{mod_name}_{b}_radar_metrics.png"), f"{mod_name} ({b.upper()}): Performance Radar")
                
                ranks = eff.get('per_layer_ranks', {})
                if ranks:
                    plot_ranks(ranks, os.path.join(b_dir, f"{b}_layer_ranks.png"), f"{mod_name} ({b.upper()}): Layer Rank Distribution")
                    plot_ranks(ranks, os.path.join(d, f"{mod_name}_{b}_layer_ranks.png"), f"{mod_name} ({b.upper()}): Layer Rank Distribution")
            except Exception:
                pass
    processed += 1

print(f"Successfully generated all diagnostic plots across {processed} module directories!")
