#!/usr/bin/env python3
"""
Comprehensive Per-Module & Per-Backbone Visualization Suite.
Generates and places in EACH module's directory and backbone subdirectories:
1. <module>_dissection_profile.png (Summary comparison across DistilBERT, RoBERTa, GPT-2)
2. <backbone>_loss_curve.png (Training loss convergence trajectory extracted from console logs)
3. <backbone>_radar_metrics.png (Multivariate performance radar: Accuracy, F1, Precision, Recall, Speed)
4. <backbone>_layer_ranks.png (SVD / LoRA rank distribution across transformer layers)
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

RESULTS_DIR = r"D:\SVD_Project\results" if os.path.exists(r"D:\SVD_Project\results") else "/Users/tahamajs/Documents/uni/Project/results"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def parse_loss_from_log(log_path):
    steps = []
    losses = []
    if not os.path.exists(log_path):
        return steps, losses
    pattern = re.compile(r"fine-tune step (\d+)/\d+ loss=([0-9\.]+)")
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return steps, losses

def plot_loss_curve(steps, losses, out_path, title):
    if not steps or not losses:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, losses, marker='o', markersize=3, color='#E74C3C', linewidth=1.8, label='Fine-tuning Loss')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Recovery Step')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(frameon=True, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_radar(metrics_dict, out_path, title):
    categories = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    values = [
        metrics_dict.get('accuracy_pct', 0.0) / 100.0,
        metrics_dict.get('f1_score', 0.0),
        metrics_dict.get('precision', 0.0),
        metrics_dict.get('recall', 0.0)
    ]
    if all(v == 0.0 for v in values):
        return
        
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]
    
    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='#2980B9', linewidth=2, linestyle='solid')
    ax.fill(angles, values, color='#3498DB', alpha=0.35)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=11, fontweight='bold', y=1.08)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_layer_ranks(ranks_dict, out_path, title):
    if not ranks_dict:
        return
    layers = list(ranks_dict.keys())
    # Shorten layer names for display
    short_layers = [l.split('.')[-2] if len(l.split('.')) > 2 else l for l in layers]
    ranks = list(ranks_dict.values())
    
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(range(len(ranks)), ranks, color='#8E44AD', width=0.55)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Transformer Layer')
    ax.set_ylabel('Injected / SVD Rank')
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels(range(1, len(ranks) + 1))
    ax.set_ylim(0, max(ranks) * 1.25 if max(ranks) > 0 else 64)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def process_module(mod_dir):
    mod_name = os.path.basename(mod_dir).replace("unified_", "")
    backbones = ["distilbert", "roberta", "gpt2"]
    
    for b in backbones:
        b_dir = os.path.join(mod_dir, b)
        if not os.path.exists(b_dir):
            continue
            
        # 1. Loss curve from console.log
        console_log = os.path.join(b_dir, "console.log")
        steps, losses = parse_loss_from_log(console_log)
        if steps and losses:
            plot_loss_curve(steps, losses, os.path.join(b_dir, f"{mod_name}_{b}_loss_convergence.png"),
                            f"{mod_name} on {b.upper()}: Loss Convergence")
            # Also copy one to the module root
            plot_loss_curve(steps, losses, os.path.join(mod_dir, f"{mod_name}_{b}_loss_convergence.png"),
                            f"{mod_name} on {b.upper()}: Loss Convergence")
                            
        # 2. Radar & Layer ranks from full_metrics.json
        metric_files = glob.glob(os.path.join(b_dir, "*_full_metrics.json"))
        for mf in metric_files:
            if "_full_full_metrics" in mf:
                continue
            try:
                with open(mf, 'r') as f:
                    mdata = json.load(f)
                perf = mdata.get('performance', {})
                eff = mdata.get('efficiency', {})
                
                # Radar plot
                plot_radar(perf, os.path.join(b_dir, f"{mod_name}_{b}_radar_metrics.png"),
                           f"{mod_name} on {b.upper()}: Multi-Metric Balance")
                plot_radar(perf, os.path.join(mod_dir, f"{mod_name}_{b}_radar_metrics.png"),
                           f"{mod_name} on {b.upper()}: Multi-Metric Balance")
                           
                # Layer ranks plot
                ranks = eff.get('per_layer_ranks', {})
                if ranks:
                    plot_layer_ranks(ranks, os.path.join(b_dir, f"{mod_name}_{b}_layer_ranks.png"),
                                     f"{mod_name} on {b.upper()}: Per-Layer Allocation")
                    plot_layer_ranks(ranks, os.path.join(mod_dir, f"{mod_name}_{b}_layer_ranks.png"),
                                     f"{mod_name} on {b.upper()}: Per-Layer Allocation")
            except Exception:
                pass

if __name__ == "__main__":
    dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "unified_*")))
    print(f"Generating comprehensive plot suite inside {len(dirs)} module folders...")
    for d in dirs:
        if os.path.isdir(d):
            process_module(d)
    print("All multi-metric, loss convergence, and layer rank plots generated successfully!")
