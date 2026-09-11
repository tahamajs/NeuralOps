#!/usr/bin/env python3
"""
Comprehensive Chart Generator — Thesis v2
Generates 18+ publication-quality figures from real benchmark data.
"""
import json, os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as mticker
from collections import defaultdict

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = '/Users/tahamajs/Documents/uni/Project'
RESULTS = os.path.join(BASE, 'results/master_aggregated_results.json')
OUT     = os.path.join(BASE, 'thesis_final/img')
os.makedirs(OUT, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
})

PALETTE = {
    'PASS':      '#2ECC71',
    'PASS_LOW':  '#F39C12',
    'FAIL':      '#E74C3C',
    'distilbert':'#3498DB',
    'gpt2':      '#9B59B6',
    'roberta':   '#E67E22',
    'qwen':      '#1ABC9C',
}

# ── Load Data ──────────────────────────────────────────────────────────────────
with open(RESULTS) as f:
    raw = json.load(f)

mods = raw['module_summary']        # dict  module_name → {...}
recs = raw.get('all_records', [])   # list  of individual records

# Flatten to list of dicts for easy filtering
modules = []
for name, v in mods.items():
    v['module_name'] = name
    # infer family from name heuristics
    n = name.lower()
    if any(x in n for x in ['svd','aasvd','dobi','cur_','direct','cure','admm','qsvd','ttsvd','eva_svd']):
        family = 'SVD/Low-Rank'
    elif any(x in n for x in ['ssm','mamba','rwkv','hgrn','recurrent','linear_rnn','lfad','s4','tnn','retention']):
        family = 'SSM/Recurrent'
    elif any(x in n for x in ['sparse','nsa','lsh','block_sparse','routing','topk','big_bird','longformer']):
        family = 'Sparse/Window'
    elif any(x in n for x in ['quant','spectral','huffman','fft','wavelet','cosine','oats','nli']):
        family = 'Quantized/Spectral'
    elif any(x in n for x in ['hybrid','ensemble','moe','mixture','adaptive']):
        family = 'Hybrid/Ensemble'
    else:
        family = 'Advanced Attention'
    v['family'] = family
    # normalise accuracy fields
    acc = v.get('accuracies', {})
    v['acc_distilbert'] = acc.get('distilbert', None)
    v['acc_gpt2']       = acc.get('gpt2',       None)
    v['acc_roberta']    = acc.get('roberta',     None)
    v['acc_qwen']       = acc.get('qwen',        None)
    v['avg_acc']        = v.get('avg_accuracy',  None)
    v['max_acc']        = v.get('max_accuracy',  None)
    modules.append(v)

# sort by avg accuracy desc
modules.sort(key=lambda x: x['avg_acc'] or 0, reverse=True)

def save(name):
    path = os.path.join(OUT, name)
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close('all')
    print(f'  ✔ saved {name}')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1: Top-30 Modules — Average Accuracy (horizontal bar)
# ══════════════════════════════════════════════════════════════════════════════
print('[1] Top-30 Average Accuracy')
top30 = [m for m in modules if m['avg_acc'] is not None][:30]
names = [m['module_name'].replace('_', ' ') for m in top30]
accs  = [m['avg_acc'] for m in top30]
colors= [PALETTE.get(m['family'], '#95A5A6') for m in top30]

fig, ax = plt.subplots(figsize=(12, 10))
bars = ax.barh(range(len(top30)), accs, color=colors, edgecolor='white', linewidth=0.5)
ax.axvline(91.2, color='red', linestyle='--', lw=1.5, label='Baseline (Full Attn 91.2%)')
for i, (bar, v) in enumerate(zip(bars, accs)):
    ax.text(v + 0.3, bar.get_y() + bar.get_height()/2, f'{v:.1f}%', va='center', fontsize=8)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(names, fontsize=8.5)
ax.set_xlabel('میانگین دقت (%)')
ax.set_title('۳۰ ماژول برتر بر اساس میانگین دقت در ۳ مدل', fontsize=14, fontweight='bold', pad=12)
ax.legend(loc='lower right')
families = list({m['family'] for m in top30})
handles = [mpatches.Patch(color=PALETTE.get(f,'#95A5A6'), label=f) for f in sorted(families)]
ax.legend(handles=handles + [plt.Line2D([0],[0],color='red',linestyle='--',lw=1.5,label='خط مبنا')],
          loc='lower right', fontsize=8)
ax.set_xlim(0, 100)
ax.invert_yaxis()
plt.tight_layout()
save('chart_top30_avg_accuracy.png')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2: Pass/Fail/Pass-Low Distribution (stacked bar per family)
# ══════════════════════════════════════════════════════════════════════════════
print('[2] Pass/Fail distribution per family')
families_order = ['SVD/Low-Rank','SSM/Recurrent','Sparse/Window','Quantized/Spectral','Hybrid/Ensemble','Advanced Attention']
fam_data = defaultdict(lambda: {'PASS':0,'PASS_LOW':0,'FAIL':0,'total':0})
for m in modules:
    fam = m['family']
    statuses = m.get('statuses', {})
    for model, st in statuses.items():
        if st in ('PASS','PASS_LOW','FAIL'):
            fam_data[fam][st] += 1
        fam_data[fam]['total'] += 1

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(families_order))
w = 0.55
bottoms = np.zeros(len(families_order))
for status, color in [('PASS','#2ECC71'),('PASS_LOW','#F39C12'),('FAIL','#E74C3C')]:
    vals = [fam_data[f][status] for f in families_order]
    bars = ax.bar(x, vals, w, bottom=bottoms, color=color, label=status, edgecolor='white')
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_y()+bar.get_height()/2,
                    str(val), ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    bottoms += np.array(vals, dtype=float)

ax.set_xticks(x)
ax.set_xticklabels([f.replace('/',' /\n') for f in families_order], fontsize=9)
ax.set_ylabel('تعداد آزمایش‌ها')
ax.set_title('توزیع نتایج PASS/FAIL بر اساس خانواده معماری', fontsize=14, fontweight='bold', pad=12)
ax.legend(title='وضعیت', loc='upper right')
plt.tight_layout()
save('chart_pass_fail_by_family.png')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 3: Heatmap — accuracy per module (top 40) × model
# ══════════════════════════════════════════════════════════════════════════════
print('[3] Accuracy heatmap module × model')
top40 = [m for m in modules if m['avg_acc'] is not None][:40]
model_cols = ['distilbert','gpt2','roberta']
heat = np.full((len(top40), 3), np.nan)
for i, m in enumerate(top40):
    for j, col in enumerate(model_cols):
        v = m.get(f'acc_{col}', None)
        if v is not None:
            heat[i,j] = v

fig, ax = plt.subplots(figsize=(8, 14))
cmap = LinearSegmentedColormap.from_list('rg', ['#E74C3C','#F39C12','#2ECC71'])
im = ax.imshow(heat, cmap=cmap, vmin=50, vmax=95, aspect='auto')
ax.set_xticks([0,1,2])
ax.set_xticklabels(['DistilBERT','GPT-2','RoBERTa'], fontsize=10, fontweight='bold')
ax.set_yticks(range(len(top40)))
ax.set_yticklabels([m['module_name'].replace('_',' ') for m in top40], fontsize=7.5)
for i in range(len(top40)):
    for j in range(3):
        if not np.isnan(heat[i,j]):
            ax.text(j, i, f'{heat[i,j]:.1f}', ha='center', va='center', fontsize=6.5,
                    color='black' if heat[i,j]>65 else 'white')
cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cbar.set_label('دقت (%)')
ax.set_title('نقشه حرارتی دقت: ۴۰ ماژول برتر × ۳ مدل', fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
save('chart_heatmap_accuracy.png')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 4: Scatter — avg accuracy vs number of models run
# ══════════════════════════════════════════════════════════════════════════════
print('[4] Scatter: accuracy vs models run')
fig, ax = plt.subplots(figsize=(9, 6))
for m in modules:
    if m['avg_acc'] is None: continue
    n = m.get('models_run', 1)
    fam = m['family']
    ax.scatter(n + np.random.uniform(-0.15, 0.15), m['avg_acc'],
               color=PALETTE.get(fam,'#95A5A6'), alpha=0.7, s=45, edgecolors='white', lw=0.5)
ax.axhline(91.2, color='red', linestyle='--', lw=1.5, label='Baseline 91.2%')
ax.axhline(80,   color='gray', linestyle=':',  lw=1,   label='آستانه ۸۰%')
handles = [mpatches.Patch(color=PALETTE.get(f,'#95A5A6'), label=f) for f in families_order]
handles += [plt.Line2D([0],[0],color='red',ls='--',label='Baseline'),
            plt.Line2D([0],[0],color='gray',ls=':',label='آستانه ۸۰%')]
ax.legend(handles=handles, fontsize=8, ncol=2)
ax.set_xlabel('تعداد مدل‌های آزمایش‌شده')
ax.set_ylabel('میانگین دقت (%)')
ax.set_title('رابطه تعداد مدل‌های آزمایش‌شده و میانگین دقت', fontsize=13, fontweight='bold')
ax.set_xticks([1,2,3,4])
plt.tight_layout()
save('chart_accuracy_vs_models_run.png')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 5: Box plot — accuracy distribution per family
# ══════════════════════════════════════════════════════════════════════════════
print('[5] Box plot accuracy per family')
fam_accs = defaultdict(list)
for m in modules:
    for col in ['acc_distilbert','acc_gpt2','acc_roberta','acc_qwen']:
        v = m.get(col, None)
        if v is not None:
            fam_accs[m['family']].append(v)

fig, ax = plt.subplots(figsize=(11, 6))
data_plot = [fam_accs[f] for f in families_order if fam_accs[f]]
labels_plot = [f for f in families_order if fam_accs[f]]
bp = ax.boxplot(data_plot, patch_artist=True, notch=False,
                boxprops=dict(linewidth=1.5),
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker='o', markersize=3, alpha=0.5))
colors_box = ['#3498DB','#9B59B6','#E67E22','#1ABC9C','#E74C3C','#2ECC71']
for patch, col in zip(bp['boxes'], colors_box):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)
ax.axhline(91.2, color='red', linestyle='--', lw=1.5, label='Baseline 91.2%')
ax.set_xticklabels([l.replace('/',' /\n') for l in labels_plot], fontsize=9)
ax.set_ylabel('دقت (%)')
ax.set_title('توزیع دقت در هر خانواده معماری', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
save('chart_boxplot_accuracy_by_family.png')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 6: Per-model accuracy comparison — top 20 modules (grouped bar)
# ══════════════════════════════════════════════════════════════════════════════
print('[6] Grouped bar: per-model accuracy, top 20')
top20 = [m for m in modules if m['avg_acc'] is not None][:20]
x = np.arange(len(top20))
w = 0.25
fig, ax = plt.subplots(figsize=(14, 6))
for idx, (model, color) in enumerate([('distilbert','#3498DB'),('gpt2','#9B59B6'),('roberta','#E67E22')]):
    vals = [m.get(f'acc_{model}', 0) or 0 for m in top20]
    bars = ax.bar(x + idx*w, vals, w, color=color, label=model.capitalize(), edgecolor='white')
ax.axhline(91.2, color='red', linestyle='--', lw=1.2, label='Baseline 91.2%')
ax.set_xticks(x + w)
ax.set_xticklabels([m['module_name'].replace('_',' ') for m in top20], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('دقت (%)')
ax.set_title('مقایسه دقت ۲۰ ماژول برتر به تفکیک مدل زبانی', fontsize=13, fontweight='bold')
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
save('chart_per_model_top20.png')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 7: Radar / Polar — family-level multi-metric comparison (simulated)
# ══════════════════════════════════════════════════════════════════════════════
print('[7] Radar chart per family')
# Compute family avg accuracy per model
fam_model_acc = defaultdict(lambda: defaultdict(list))
for m in modules:
    fam = m['family']
    for model in ['distilbert','gpt2','roberta']:
        v = m.get(f'acc_{model}', None)
        if v is not None:
            fam_model_acc[fam][model].append(v)

# Metrics: avg_distilbert, avg_gpt2, avg_roberta, pass_rate, module_count(normalised)
metrics = ['DistilBERT', 'GPT-2', 'RoBERTa', 'نرخ PASS', 'تعداد ماژول']
N = len(metrics)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fam_pass = defaultdict(lambda: [0,0])  # [pass, total]
for m in modules:
    fam = m['family']
    for model, st in m.get('statuses',{}).items():
        fam_pass[fam][1] += 1
        if st == 'PASS':
            fam_pass[fam][0] += 1

fam_count = defaultdict(int)
for m in modules:
    fam_count[m['family']] += 1
max_count = max(fam_count.values()) or 1

fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
radar_colors = ['#3498DB','#9B59B6','#E67E22','#1ABC9C','#E74C3C','#2ECC71']
for fam, col in zip(families_order, radar_colors):
    db  = np.mean(fam_model_acc[fam]['distilbert']) if fam_model_acc[fam]['distilbert'] else 50
    gpt = np.mean(fam_model_acc[fam]['gpt2'])       if fam_model_acc[fam]['gpt2']       else 50
    rob = np.mean(fam_model_acc[fam]['roberta'])     if fam_model_acc[fam]['roberta']    else 50
    pr  = (fam_pass[fam][0]/fam_pass[fam][1]*100)   if fam_pass[fam][1]                 else 0
    cnt = fam_count[fam]/max_count*100
    vals = [db, gpt, rob, pr, cnt]
    vals += vals[:1]
    ax.plot(angles, vals, color=col, linewidth=2, label=fam)
    ax.fill(angles, vals, color=col, alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylim(0, 100)
ax.set_title('مقایسه رادار خانواده‌های معماری', fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
plt.tight_layout()
save('chart_radar_families.png')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 8: Histogram — distribution of average accuracy across all modules
# ══════════════════════════════════════════════════════════════════════════════
print('[8] Histogram: accuracy distribution')
all_accs = [m['avg_acc'] for m in modules if m['avg_acc'] is not None]
fig, ax = plt.subplots(figsize=(9,5))
n, bins, patches = ax.hist(all_accs, bins=20, edgecolor='white', color='#3498DB', alpha=0.8)
# color by range
for patch, left in zip(patches, bins):
    if left >= 85:
        patch.set_facecolor('#2ECC71')
    elif left >= 75:
        patch.set_facecolor('#F39C12')
    elif left >= 60:
        patch.set_facecolor('#E67E22')
    else:
        patch.set_facecolor('#E74C3C')
ax.axvline(np.mean(all_accs), color='navy', linestyle='--', lw=2, label=f'میانگین: {np.mean(all_accs):.1f}%')
ax.axvline(91.2, color='red', linestyle='--', lw=1.5, label='Baseline 91.2%')
ax.set_xlabel('میانگین دقت (%)')
ax.set_ylabel('تعداد ماژول')
ax.set_title('توزیع دقت در ۱۵۰ ماژول بررسی‌شده', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
save('chart_accuracy_histogram.png')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 9: Stacked area — cumulative modules reaching accuracy thresholds
# ══════════════════════════════════════════════════════════════════════════════
print('[9] Cumulative accuracy threshold')
thresholds = np.arange(50, 95, 2)
cum = [sum(1 for m in modules if (m['avg_acc'] or 0) >= t) for t in thresholds]
fig, ax = plt.subplots(figsize=(9,5))
ax.fill_between(thresholds, cum, alpha=0.4, color='#3498DB')
ax.plot(thresholds, cum, color='#2C3E50', lw=2.5)
for t in [80, 85, 90]:
    c = sum(1 for m in modules if (m['avg_acc'] or 0) >= t)
    ax.annotate(f'{c} ماژول\n≥{t}%', xy=(t,c), xytext=(t+1, c+3),
                arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9, color='#2C3E50')
ax.set_xlabel('آستانه دقت (%)')
ax.set_ylabel('تعداد ماژول‌های برتر از آستانه')
ax.set_title('تعداد ماژول‌ها به ازای هر آستانه دقت', fontsize=13, fontweight='bold')
plt.tight_layout()
save('chart_cumulative_threshold.png')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 10: Per-model mean accuracy comparison (bar) + error bars
# ══════════════════════════════════════════════════════════════════════════════
print('[10] Model-level mean accuracy')
model_list = ['distilbert','gpt2','roberta']
model_accs = {}
for model in model_list:
    vals = [m.get(f'acc_{model}') for m in modules if m.get(f'acc_{model}') is not None]
    model_accs[model] = (np.mean(vals), np.std(vals), len(vals))

fig, ax = plt.subplots(figsize=(7,5))
colors_m = ['#3498DB','#9B59B6','#E67E22']
xlabels = ['DistilBERT','GPT-2','RoBERTa']
means = [model_accs[m][0] for m in model_list]
stds  = [model_accs[m][1] for m in model_list]
counts= [model_accs[m][2] for m in model_list]
bars = ax.bar(xlabels, means, color=colors_m, edgecolor='white', linewidth=0.8, width=0.5)
ax.errorbar(xlabels, means, yerr=stds, fmt='none', color='#2C3E50', capsize=6, lw=2)
ax.axhline(91.2, color='red', linestyle='--', lw=1.5, label='Baseline 91.2%')
for bar, mean, std, cnt in zip(bars, means, stds, counts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+std+1,
            f'{mean:.1f}%\n(n={cnt})', ha='center', fontsize=9)
ax.set_ylabel('میانگین دقت (%)')
ax.set_ylim(0, 105)
ax.set_title('میانگین دقت به تفکیک مدل زبانی پایه', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
save('chart_model_mean_accuracy.png')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 11: Family count donut chart
# ══════════════════════════════════════════════════════════════════════════════
print('[11] Family count donut')
fam_counts = {f: fam_count[f] for f in families_order if fam_count[f] > 0}
fig, ax = plt.subplots(figsize=(8,6))
sizes  = list(fam_counts.values())
labels = [f'{k}\n({v} ماژول)' for k,v in fam_counts.items()]
explode = [0.03]*len(sizes)
wedge_colors = ['#3498DB','#9B59B6','#E67E22','#1ABC9C','#E74C3C','#2ECC71']
wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                   autopct='%1.1f%%', startangle=140,
                                   colors=wedge_colors, pctdistance=0.78,
                                   wedgeprops=dict(width=0.5, edgecolor='white', linewidth=1.5))
for t in autotexts:
    t.set_fontsize(8.5)
    t.set_fontweight('bold')
ax.set_title('توزیع ماژول‌ها بر اساس خانواده معماری', fontsize=13, fontweight='bold')
plt.tight_layout()
save('chart_family_donut.png')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 12: Top vs Bottom 15 — side-by-side bar
# ══════════════════════════════════════════════════════════════════════════════
print('[12] Top vs Bottom 15')
valid = [m for m in modules if m['avg_acc'] is not None]
top15   = valid[:15]
bottom15= valid[-15:]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
for ax, grp, title, col in [
    (ax1, top15,    '۱۵ ماژول برتر', '#2ECC71'),
    (ax2, bottom15, '۱۵ ماژول ضعیف‌تر', '#E74C3C')]:
    names = [m['module_name'].replace('_',' ') for m in grp]
    accs  = [m['avg_acc'] for m in grp]
    bars  = ax.barh(range(len(grp)), accs, color=col, alpha=0.8, edgecolor='white')
    ax.axvline(91.2, color='red', linestyle='--', lw=1.2)
    ax.axvline(80, color='gray', linestyle=':', lw=1)
    for bar, v in zip(bars, accs):
        ax.text(v+0.5, bar.get_y()+bar.get_height()/2, f'{v:.1f}%', va='center', fontsize=8)
    ax.set_yticks(range(len(grp)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('میانگین دقت (%)')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
plt.suptitle('مقایسه بهترین و ضعیف‌ترین ماژول‌ها', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save('chart_top_vs_bottom15.png')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 13: RoBERTa vs DistilBERT accuracy scatter
# ══════════════════════════════════════════════════════════════════════════════
print('[13] RoBERTa vs DistilBERT scatter')
x_vals, y_vals, cols, names_list = [], [], [], []
for m in modules:
    r = m.get('acc_roberta', None)
    d = m.get('acc_distilbert', None)
    if r is not None and d is not None:
        x_vals.append(d)
        y_vals.append(r)
        cols.append(PALETTE.get(m['family'],'#95A5A6'))
        names_list.append(m['module_name'])

fig, ax = plt.subplots(figsize=(8,7))
ax.scatter(x_vals, y_vals, c=cols, alpha=0.7, s=60, edgecolors='white', lw=0.5)
lim = (45, 100)
ax.plot(lim, lim, 'k--', lw=1, alpha=0.4, label='خط برابری y=x')
ax.axhline(91.2, color='red',  linestyle=':', lw=1, alpha=0.6)
ax.axvline(91.2, color='blue', linestyle=':', lw=1, alpha=0.6)
# annotate top performers
for x, y, name in zip(x_vals, y_vals, names_list):
    if x > 85 or y > 88:
        ax.annotate(name.replace('_',' '), (x,y), fontsize=6.5, alpha=0.8,
                    xytext=(3,3), textcoords='offset points')
ax.set_xlabel('دقت روی DistilBERT (%)')
ax.set_ylabel('دقت روی RoBERTa (%)')
ax.set_title('مقایسه دقت DistilBERT در برابر RoBERTa', fontsize=13, fontweight='bold')
handles = [mpatches.Patch(color=PALETTE.get(f,'#95A5A6'), label=f) for f in families_order]
ax.legend(handles=handles, fontsize=7.5, loc='lower right', ncol=2)
plt.tight_layout()
save('chart_roberta_vs_distilbert.png')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 14: GPT-2 vs RoBERTa scatter
# ══════════════════════════════════════════════════════════════════════════════
print('[14] GPT-2 vs RoBERTa scatter')
x_vals, y_vals, cols = [], [], []
for m in modules:
    g = m.get('acc_gpt2', None)
    r = m.get('acc_roberta', None)
    if g is not None and r is not None:
        x_vals.append(g)
        y_vals.append(r)
        cols.append(PALETTE.get(m['family'],'#95A5A6'))

fig, ax = plt.subplots(figsize=(8,7))
ax.scatter(x_vals, y_vals, c=cols, alpha=0.7, s=60, edgecolors='white', lw=0.5)
lim = (45, 100)
ax.plot(lim, lim, 'k--', lw=1, alpha=0.4, label='خط برابری')
ax.set_xlabel('دقت روی GPT-2 (%)')
ax.set_ylabel('دقت روی RoBERTa (%)')
ax.set_title('مقایسه دقت GPT-2 در برابر RoBERTa', fontsize=13, fontweight='bold')
handles = [mpatches.Patch(color=PALETTE.get(f,'#95A5A6'), label=f) for f in families_order]
ax.legend(handles=handles+[plt.Line2D([0],[0],ls='--',color='k',label='y=x')], fontsize=7.5, loc='lower right')
plt.tight_layout()
save('chart_gpt2_vs_roberta.png')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 15: Module pass rate by family (horizontal bar)
# ══════════════════════════════════════════════════════════════════════════════
print('[15] Pass rate per family')
fam_pass_rate = {}
for fam in families_order:
    p, t = fam_pass[fam]
    fam_pass_rate[fam] = (p/t*100) if t > 0 else 0

fig, ax = plt.subplots(figsize=(9,5))
fams_sorted = sorted(fam_pass_rate, key=fam_pass_rate.get, reverse=True)
rates = [fam_pass_rate[f] for f in fams_sorted]
bar_colors = [radar_colors[families_order.index(f)] for f in fams_sorted]
bars = ax.barh(fams_sorted, rates, color=bar_colors, edgecolor='white', alpha=0.85)
for bar, rate in zip(bars, rates):
    ax.text(rate+0.5, bar.get_y()+bar.get_height()/2, f'{rate:.1f}%', va='center', fontsize=9)
ax.axvline(50, color='gray', ls=':', lw=1, label='۵۰٪ آستانه پایه')
ax.set_xlabel('نرخ PASS (%)')
ax.set_title('نرخ موفقیت (PASS) به تفکیک خانواده معماری', fontsize=13, fontweight='bold')
ax.set_xlim(0, 110)
ax.legend()
plt.tight_layout()
save('chart_pass_rate_by_family.png')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 16: Violin plot — accuracy distribution per model
# ══════════════════════════════════════════════════════════════════════════════
print('[16] Violin: accuracy per model')
model_vals_all = {}
for model in ['distilbert','gpt2','roberta']:
    vals = [m.get(f'acc_{model}') for m in modules if m.get(f'acc_{model}') is not None]
    if vals:
        model_vals_all[model] = vals

fig, ax = plt.subplots(figsize=(8,6))
positions = list(range(1, len(model_vals_all)+1))
labels_v = list(model_vals_all.keys())
data_v   = [model_vals_all[l] for l in labels_v]
parts = ax.violinplot(data_v, positions=positions, showmedians=True, showextrema=True)
vcolors = ['#3498DB','#9B59B6','#E67E22']
for i, (pc, col) in enumerate(zip(parts['bodies'], vcolors)):
    pc.set_facecolor(col)
    pc.set_alpha(0.7)
parts['cmedians'].set_color('black')
parts['cmedians'].set_linewidth(2)
ax.axhline(91.2, color='red', ls='--', lw=1.5, label='Baseline 91.2%')
ax.set_xticks(positions)
ax.set_xticklabels(['DistilBERT','GPT-2','RoBERTa'], fontsize=11)
ax.set_ylabel('دقت (%)')
ax.set_title('توزیع دقت به تفکیک مدل زبانی (Violin Plot)', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
save('chart_violin_accuracy.png')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 17: Summary dashboard — 4-panel overview
# ══════════════════════════════════════════════════════════════════════════════
print('[17] 4-panel dashboard')
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

# Panel A: accuracy histogram
ax_a = fig.add_subplot(gs[0, 0])
ax_a.hist(all_accs, bins=15, color='#3498DB', edgecolor='white', alpha=0.8)
ax_a.axvline(np.mean(all_accs), color='navy', ls='--', lw=2, label=f'μ={np.mean(all_accs):.1f}%')
ax_a.axvline(91.2, color='red', ls='--', lw=1.5, label='BL=91.2%')
ax_a.set_title('الف) توزیع دقت کلی', fontweight='bold')
ax_a.set_xlabel('دقت (%)')
ax_a.set_ylabel('تعداد')
ax_a.legend(fontsize=8)

# Panel B: pass/fail pie
ax_b = fig.add_subplot(gs[0, 1])
total_pass = sum(1 for m in modules for st in m.get('statuses',{}).values() if st=='PASS')
total_low  = sum(1 for m in modules for st in m.get('statuses',{}).values() if st=='PASS_LOW')
total_fail = sum(1 for m in modules for st in m.get('statuses',{}).values() if st=='FAIL')
pf_vals = [total_pass, total_low, total_fail]
pf_cols = ['#2ECC71','#F39C12','#E74C3C']
pf_lbls = [f'PASS\n({total_pass})', f'PASS_LOW\n({total_low})', f'FAIL\n({total_fail})']
ax_b.pie(pf_vals, labels=pf_lbls, colors=pf_cols, autopct='%1.0f%%',
         wedgeprops=dict(width=0.55, edgecolor='white',lw=1.5), pctdistance=0.75, startangle=90)
ax_b.set_title('ب) توزیع کلی وضعیت آزمایش‌ها', fontweight='bold')

# Panel C: per-model bar
ax_c = fig.add_subplot(gs[1, 0])
means_c = [np.mean([m.get(f'acc_{mo}') for m in modules if m.get(f'acc_{mo}')]) for mo in model_list]
ax_c.bar(['DistilBERT','GPT-2','RoBERTa'], means_c, color=['#3498DB','#9B59B6','#E67E22'], edgecolor='white', width=0.5)
ax_c.axhline(91.2, color='red', ls='--', lw=1.5, label='BL=91.2%')
for i, v in enumerate(means_c):
    ax_c.text(i, v+0.5, f'{v:.1f}%', ha='center', fontsize=9)
ax_c.set_ylabel('میانگین دقت (%)')
ax_c.set_ylim(0, 100)
ax_c.set_title('ج) میانگین دقت به تفکیک مدل', fontweight='bold')
ax_c.legend(fontsize=8)

# Panel D: top-10 modules
ax_d = fig.add_subplot(gs[1, 1])
top10 = valid[:10]
ax_d.barh(range(10), [m['avg_acc'] for m in top10],
          color=['#2ECC71']*10, edgecolor='white', alpha=0.85)
ax_d.set_yticks(range(10))
ax_d.set_yticklabels([m['module_name'].replace('_',' ') for m in top10], fontsize=8)
ax_d.set_xlabel('میانگین دقت (%)')
ax_d.set_title('د) ۱۰ ماژول برتر', fontweight='bold')
ax_d.invert_yaxis()
ax_d.set_xlim(0, 100)

plt.suptitle('داشبورد جامع نتایج ارزیابی ۱۵۰ ماژول', fontsize=15, fontweight='bold', y=1.01)
save('chart_dashboard_4panel.png')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 18: Top family mean accuracy (lollipop)
# ══════════════════════════════════════════════════════════════════════════════
print('[18] Lollipop family mean accuracy')
fam_avg_acc = {}
for fam in families_order:
    vals = [m['avg_acc'] for m in modules if m['family']==fam and m['avg_acc'] is not None]
    if vals:
        fam_avg_acc[fam] = np.mean(vals)

fig, ax = plt.subplots(figsize=(9,5))
fams_s = sorted(fam_avg_acc, key=fam_avg_acc.get, reverse=True)
xs = range(len(fams_s))
ys = [fam_avg_acc[f] for f in fams_s]
ax.hlines(ys, xmin=-0.4, xmax=xs, linestyle='--', color='gray', alpha=0.4)
ax.plot(xs, ys, 'o', markersize=14, color='#3498DB', zorder=5)
for x, y, fam in zip(xs, ys, fams_s):
    ax.text(x, y+0.8, f'{y:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax.axhline(91.2, color='red', ls='--', lw=1.5, label='Baseline 91.2%')
ax.set_xticks(xs)
ax.set_xticklabels([f.replace('/',' /\n') for f in fams_s], fontsize=9)
ax.set_ylabel('میانگین دقت (%)')
ax.set_ylim(50, 100)
ax.set_title('میانگین دقت به تفکیک خانواده معماری', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
save('chart_lollipop_family.png')

print('\n✅ All 18 charts generated successfully.')
print(f'   Output dir: {OUT}')
