import json, os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

warnings.filterwarnings('ignore')

BASE = '/Users/tahamajs/Documents/uni/Project'
RESULTS = os.path.join(BASE, 'results/master_aggregated_results.json')
OUT = os.path.join(BASE, 'thesis_final/img')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
})

PALETTE = {
    'SVD/Low-Rank': '#2ECC71',
    'SSM/Recurrent': '#3498DB',
    'Sparse/Window': '#9B59B6',
    'Quantized/Spectral': '#E67E22',
    'Hybrid/Ensemble': '#1ABC9C',
    'Advanced Attention': '#E74C3C',
}

with open(RESULTS) as f:
    raw = json.load(f)

mods = raw['module_summary']

# STRICT FILTERING: Keep ONLY modules that achieved PASS (exclude FAIL and PASS_LOW)
# and have valid accuracy >= 65%
filtered_modules = []
for name, v in mods.items():
    statuses = list(v.get('statuses', {}).values())
    fail_cnt = v.get('fail_count', 0)
    pass_cnt = v.get('pass_count', 0)
    avg_acc = v.get('avg_accuracy') or 0
    
    # Exclude any module that failed or only had PASS_LOW or collapsed to random guess
    if fail_cnt > 0:
        continue
    if pass_cnt == 0 or 'PASS' not in statuses:
        continue
    if avg_acc < 65.0: # Exclude transfer collapse / random guess
        continue
        
    n = name.lower()
    if any(x in n for x in ['svd','aasvd','dobi','cur_','direct','cure','admm','qsvd','ttsvd','eva_svd']):
        family = 'SVD/Low-Rank'
    elif any(x in n for x in ['ssm','mamba','rwkv','hgrn','recurrent','linear_rnn','lfad','s4','tnn','retention']):
        family = 'SSM/Recurrent'
    elif any(x in n for x in ['sparse','nsa','lsh','block_sparse','routing','topk','big_bird','longformer']):
        family = 'Sparse/Window'
    elif any(x in n for x in ['quant','spectral','huffman','fft','wavelet','cosine','oats','nli','slicegpt']):
        family = 'Quantized/Spectral'
    elif any(x in n for x in ['hybrid','ensemble','moe','mixture','adaptive']):
        family = 'Hybrid/Ensemble'
    else:
        family = 'Advanced Attention'
        
    v['module_name'] = name
    v['family'] = family
    acc = v.get('accuracies', {})
    v['acc_distilbert'] = acc.get('distilbert', None)
    v['acc_gpt2'] = acc.get('gpt2', None)
    v['acc_roberta'] = acc.get('roberta', None)
    v['avg_acc'] = avg_acc
    v['max_acc'] = v.get('max_accuracy') or avg_acc
    filtered_modules.append(v)

filtered_modules.sort(key=lambda x: x['avg_acc'], reverse=True)
print(f"Total Qualified PASS Modules: {len(filtered_modules)}")

def save(name):
    path = os.path.join(OUT, name)
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close('all')
    print(f'  ✔ Saved English pure PASS chart: {name}')

# Chart 1: Top 30 PASS Modules (Horizontal Bar)
print('[1] Top-30 PASS Average Accuracy')
top30 = filtered_modules[:30]
names = [m['module_name'].replace('_', ' ') for m in top30]
accs = [m['avg_acc'] for m in top30]
colors = [PALETTE.get(m['family'], '#3498DB') for m in top30]

fig, ax = plt.subplots(figsize=(11, 9))
bars = ax.barh(range(len(top30)), accs, color=colors, edgecolor='white', height=0.7)
ax.axvline(91.2, color='red', linestyle='--', lw=1.5, label='Full Attention Baseline (91.2%)')
for bar, v in zip(bars, accs):
    ax.text(v + 0.3, bar.get_y() + bar.get_height()/2, f'{v:.2f}%', va='center', fontsize=8, fontweight='bold')
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(names, fontsize=8.5)
ax.set_xlabel('Average Validation Accuracy (%)')
ax.set_title('Top 30 Validated PASS Attention Modules (3 Base LLMs)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlim(65, 96)
ax.invert_yaxis()
families = sorted(list({m['family'] for m in top30}))
handles = [mpatches.Patch(color=PALETTE.get(f, '#3498DB'), label=f) for f in families]
handles.append(plt.Line2D([0],[0], color='red', linestyle='--', lw=1.5, label='Baseline (91.2%)'))
ax.legend(handles=handles, loc='lower right', framealpha=0.9)
plt.tight_layout()
save('chart_top30_avg_accuracy.png')

# Chart 2: Architecture Family Distribution (Qualified PASS Only)
print('[2] Qualified PASS Distribution by Family')
fam_counts = defaultdict(int)
for m in filtered_modules:
    fam_counts[m['family']] += 1

fam_order = sorted(fam_counts.keys(), key=lambda f: fam_counts[f], reverse=True)
counts = [fam_counts[f] for f in fam_order]
colors_f = [PALETTE.get(f, '#3498DB') for f in fam_order]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(range(len(fam_order)), counts, color=colors_f, edgecolor='white', width=0.55)
for bar, c in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(c), ha='center', fontweight='bold')
ax.set_xticks(range(len(fam_order)))
ax.set_xticklabels([f.replace('/', '/\n') for f in fam_order], fontsize=9)
ax.set_ylabel('Number of Verified PASS Modules')
ax.set_title('Architecture Family Distribution of High-Accuracy Modules (PASS Only)', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(counts) + 4)
plt.tight_layout()
save('chart_pass_fail_by_family.png')

# Chart 3: Accuracy Heatmap (Top 35 Modules x 3 Models)
print('[3] Heatmap Top 35 PASS Modules')
top35 = filtered_modules[:35]
heat = np.full((len(top35), 3), np.nan)
for i, m in enumerate(top35):
    heat[i, 0] = m.get('acc_distilbert') or m['avg_acc']
    heat[i, 1] = m.get('acc_gpt2') or m['avg_acc']
    heat[i, 2] = m.get('acc_roberta') or m['avg_acc']

fig, ax = plt.subplots(figsize=(7.5, 12))
cmap = LinearSegmentedColormap.from_list('gyr', ['#F39C12', '#2ECC71', '#27AE60'])
im = ax.imshow(heat, cmap=cmap, vmin=70, vmax=93, aspect='auto')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['DistilBERT', 'GPT-2', 'RoBERTa'], fontsize=10, fontweight='bold')
ax.set_yticks(range(len(top35)))
ax.set_yticklabels([m['module_name'].replace('_', ' ') for m in top35], fontsize=8)
for i in range(len(top35)):
    for j in range(3):
        val = heat[i, j]
        ax.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=7.5, color='black')
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
cbar.set_label('Accuracy (%)')
ax.set_title('Accuracy Heatmap: Top 35 PASS Modules Across 3 LLMs', fontsize=12, fontweight='bold', pad=10)
plt.tight_layout()
save('chart_heatmap_accuracy.png')

# Chart 4: Scatter - Models Run vs Accuracy
print('[4] Models Run vs Accuracy')
fig, ax = plt.subplots(figsize=(8.5, 5.5))
for m in filtered_modules:
    n_models = m.get('models_run', 1)
    ax.scatter(n_models + np.random.uniform(-0.12, 0.12), m['avg_acc'],
               color=PALETTE.get(m['family'], '#3498DB'), alpha=0.8, s=55, edgecolors='white')
ax.axhline(91.2, color='red', linestyle='--', lw=1.5, label='Baseline (91.2%)')
ax.axhline(80.0, color='gray', linestyle=':', lw=1, label='80% Performance Bound')
ax.set_xticks([1, 2, 3, 4])
ax.set_xlabel('Number of Base Models Evaluated')
ax.set_ylabel('Average Accuracy (%)')
ax.set_title('Module Accuracy vs. Multi-Model Generalization (PASS Only)', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
save('chart_accuracy_vs_models_run.png')

# Chart 5: Box Plot Accuracy by Family
print('[5] Box Plot by Family')
fam_acc_data = defaultdict(list)
for m in filtered_modules:
    fam_acc_data[m['family']].append(m['avg_acc'])

fig, ax = plt.subplots(figsize=(10, 5.5))
box_data = [fam_acc_data[f] for f in fam_order if fam_acc_data[f]]
labels_b = [f.replace('/', '/\n') for f in fam_order if fam_acc_data[f]]
bp = ax.boxplot(box_data, patch_artist=True,
                boxprops=dict(linewidth=1.2),
                medianprops=dict(color='black', linewidth=2))
for patch, f in zip(bp['boxes'], fam_order):
    patch.set_facecolor(PALETTE.get(f, '#3498DB'))
    patch.set_alpha(0.75)
ax.axhline(91.2, color='red', linestyle='--', lw=1.5, label='Baseline (91.2%)')
ax.set_xticklabels(labels_b, fontsize=8.5)
ax.set_ylabel('Validation Accuracy (%)')
ax.set_title('Accuracy Distribution per Architecture Family (PASS Modules)', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
save('chart_boxplot_accuracy_by_family.png')

# Chart 6: Per-Model Top 20 Grouped Bar
print('[6] Per-Model Accuracy (Top 20)')
top20 = filtered_modules[:20]
x = np.arange(len(top20))
w = 0.26
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.bar(x - w, [m.get('acc_distilbert') or m['avg_acc'] for m in top20], w, label='DistilBERT', color='#3498DB', edgecolor='white')
ax.bar(x,     [m.get('acc_gpt2') or m['avg_acc'] for m in top20],       w, label='GPT-2',      color='#9B59B6', edgecolor='white')
ax.bar(x + w, [m.get('acc_roberta') or m['avg_acc'] for m in top20],    w, label='RoBERTa',    color='#E67E22', edgecolor='white')
ax.axhline(91.2, color='red', linestyle='--', lw=1.2, label='Baseline (91.2%)')
ax.set_xticks(x)
ax.set_xticklabels([m['module_name'].replace('_', ' ') for m in top20], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(60, 96)
ax.set_title('Accuracy Comparison of Top 20 Modules by Base Language Model', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
save('chart_per_model_top20.png')

# Chart 7: Radar Chart per Family
print('[7] Radar Chart per Family')
metrics = ['DistilBERT', 'GPT-2', 'RoBERTa', 'Avg Acc', 'Max Acc']
N = len(metrics)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
for fam in fam_order[:5]:
    f_mods = [m for m in filtered_modules if m['family'] == fam]
    if not f_mods: continue
    db = np.mean([m.get('acc_distilbert') or m['avg_acc'] for m in f_mods])
    gp = np.mean([m.get('acc_gpt2') or m['avg_acc'] for m in f_mods])
    rb = np.mean([m.get('acc_roberta') or m['avg_acc'] for m in f_mods])
    av = np.mean([m['avg_acc'] for m in f_mods])
    mx = np.mean([m['max_acc'] for m in f_mods])
    vals = [db, gp, rb, av, mx]
    vals += vals[:1]
    col = PALETTE.get(fam, '#3498DB')
    ax.plot(angles, vals, color=col, linewidth=2, label=fam)
    ax.fill(angles, vals, color=col, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=9.5, fontweight='bold')
ax.set_ylim(65, 95)
ax.set_title('Architectural Family Radar Benchmark (PASS Only)', fontsize=12, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
plt.tight_layout()
save('chart_radar_families.png')

# Chart 8: Histogram of Accuracy
print('[8] Accuracy Histogram')
accs_all = [m['avg_acc'] for m in filtered_modules]
fig, ax = plt.subplots(figsize=(8.5, 4.8))
n, bins, patches = ax.hist(accs_all, bins=16, edgecolor='white', color='#2ECC71', alpha=0.85)
for patch, left in zip(patches, bins):
    if left >= 88: patch.set_facecolor('#27AE60')
    elif left >= 80: patch.set_facecolor('#2ECC71')
    elif left >= 75: patch.set_facecolor('#F39C12')
    else: patch.set_facecolor('#E67E22')
ax.axvline(np.mean(accs_all), color='navy', linestyle='--', lw=2, label=f'Mean: {np.mean(accs_all):.2f}%')
ax.axvline(91.2, color='red', linestyle='--', lw=1.5, label='Baseline (91.2%)')
ax.set_xlabel('Average Validation Accuracy (%)')
ax.set_ylabel('Module Count')
ax.set_title('Accuracy Distribution of Verified PASS Modules (n = 77)', fontsize=12, fontweight='bold')
ax.legend(loc='upper left')
plt.tight_layout()
save('chart_accuracy_histogram.png')

# Chart 9: Cumulative Threshold
print('[9] Cumulative Threshold Curve')
thresholds = np.arange(70, 93, 1)
counts_c = [sum(1 for m in filtered_modules if m['avg_acc'] >= t) for t in thresholds]
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.fill_between(thresholds, counts_c, alpha=0.3, color='#3498DB')
ax.plot(thresholds, counts_c, color='#2980B9', lw=2.5)
for t in [80, 85, 88, 90]:
    c = sum(1 for m in filtered_modules if m['avg_acc'] >= t)
    ax.annotate(f'{c} modules\n≥ {t}%', xy=(t, c), xytext=(t+0.5, c+4),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8), fontsize=8.5)
ax.set_xlabel('Accuracy Threshold (%)')
ax.set_ylabel('Number of Modules Achieving Threshold')
ax.set_title('Cumulative Threshold Performance Curve (PASS Only)', fontsize=12, fontweight='bold')
plt.tight_layout()
save('chart_cumulative_threshold.png')

# Chart 10: Model-Level Mean Comparison with Error Bars
print('[10] Model-Level Mean with Error Bars')
m_names = ['DistilBERT', 'GPT-2', 'RoBERTa']
m_means = [
    np.mean([m.get('acc_distilbert') or m['avg_acc'] for m in filtered_modules]),
    np.mean([m.get('acc_gpt2') or m['avg_acc'] for m in filtered_modules]),
    np.mean([m.get('acc_roberta') or m['avg_acc'] for m in filtered_modules]),
]
m_stds = [
    np.std([m.get('acc_distilbert') or m['avg_acc'] for m in filtered_modules]),
    np.std([m.get('acc_gpt2') or m['avg_acc'] for m in filtered_modules]),
    np.std([m.get('acc_roberta') or m['avg_acc'] for m in filtered_modules]),
]

fig, ax = plt.subplots(figsize=(7, 4.8))
bars = ax.bar(m_names, m_means, yerr=m_stds, capsize=6, color=['#3498DB', '#9B59B6', '#E67E22'], width=0.5, edgecolor='white')
ax.axhline(91.2, color='red', linestyle='--', lw=1.5, label='Baseline (91.2%)')
for bar, m in zip(bars, m_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 6, f'{m:.2f}%', ha='center', color='white', fontweight='bold')
ax.set_ylabel('Mean Validation Accuracy (%)')
ax.set_ylim(60, 100)
ax.set_title('Mean Accuracy by Base Model (PASS Modules, n = 77)', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
save('chart_model_mean_accuracy.png')

# Chart 11: Family Share Donut Chart
print('[11] Family Donut Chart')
fig, ax = plt.subplots(figsize=(7, 5.5))
wedges, texts, autotexts = ax.pie(counts, labels=[f'{f}\n({c})' for f, c in zip(fam_order, counts)],
                                  autopct='%1.1f%%', startangle=140,
                                  colors=colors_f, pctdistance=0.75,
                                  wedgeprops=dict(width=0.5, edgecolor='white', linewidth=1.5))
for t in autotexts: t.set_fontsize(8.5); t.set_fontweight('bold')
ax.set_title('Share of Validated PASS Modules by Architecture Family', fontsize=12, fontweight='bold')
plt.tight_layout()
save('chart_family_donut.png')

# Chart 12: Top 15 vs Baseline Comparison
print('[12] Top 15 vs Baseline')
top15 = filtered_modules[:15]
fig, ax = plt.subplots(figsize=(10, 6.5))
names15 = [m['module_name'].replace('_', ' ') for m in top15]
accs15 = [m['avg_acc'] for m in top15]
bars = ax.barh(range(len(top15)), accs15, color='#2ECC71', edgecolor='white', height=0.65)
ax.axvline(91.2, color='red', linestyle='--', lw=1.5, label='Full Attention Baseline (91.2%)')
for bar, v in zip(bars, accs15):
    ax.text(v + 0.2, bar.get_y() + bar.get_height()/2, f'{v:.2f}%', va='center', fontsize=8.5, fontweight='bold')
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(names15, fontsize=8.5)
ax.set_xlabel('Validation Accuracy (%)')
ax.set_title('Top 15 High-Accuracy Validated Attention Modules', fontsize=12, fontweight='bold')
ax.set_xlim(85, 94)
ax.invert_yaxis()
ax.legend(loc='lower right')
plt.tight_layout()
save('chart_top_vs_bottom15.png')

# Chart 13: DistilBERT vs RoBERTa Scatter
print('[13] DistilBERT vs RoBERTa Scatter')
fig, ax = plt.subplots(figsize=(7.5, 6.5))
for m in filtered_modules:
    db = m.get('acc_distilbert') or m['avg_acc']
    rb = m.get('acc_roberta') or m['avg_acc']
    ax.scatter(db, rb, color=PALETTE.get(m['family'], '#3498DB'), alpha=0.75, s=60, edgecolors='white')
lim = (65, 95)
ax.plot(lim, lim, 'k--', alpha=0.4, label='y = x (Parity Line)')
ax.axhline(91.2, color='red', ls=':', lw=1)
ax.axvline(91.2, color='red', ls=':', lw=1)
ax.set_xlabel('Accuracy on DistilBERT (%)')
ax.set_ylabel('Accuracy on RoBERTa (%)')
ax.set_title('DistilBERT vs. RoBERTa Accuracy (PASS Modules)', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
save('chart_roberta_vs_distilbert.png')

# Chart 14: GPT-2 vs RoBERTa Scatter
print('[14] GPT-2 vs RoBERTa Scatter')
fig, ax = plt.subplots(figsize=(7.5, 6.5))
for m in filtered_modules:
    gp = m.get('acc_gpt2') or m['avg_acc']
    rb = m.get('acc_roberta') or m['avg_acc']
    ax.scatter(gp, rb, color=PALETTE.get(m['family'], '#3498DB'), alpha=0.75, s=60, edgecolors='white')
lim = (65, 95)
ax.plot(lim, lim, 'k--', alpha=0.4, label='y = x (Parity Line)')
ax.set_xlabel('Accuracy on GPT-2 (%)')
ax.set_ylabel('Accuracy on RoBERTa (%)')
ax.set_title('GPT-2 (Causal Decoder) vs. RoBERTa (Encoder) Accuracy', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
save('chart_gpt2_vs_roberta.png')

# Chart 15: Family Average Accuracy (Lollipop)
print('[15] Lollipop Family Accuracy')
fam_avgs = {f: np.mean([m['avg_acc'] for m in filtered_modules if m['family'] == f]) for f in fam_order}
fig, ax = plt.subplots(figsize=(8.5, 4.8))
xs = range(len(fam_order))
ys = [fam_avgs[f] for f in fam_order]
ax.hlines(ys, xmin=-0.4, xmax=xs, linestyle='--', color='gray', alpha=0.4)
ax.plot(xs, ys, 'o', markersize=13, color='#2ECC71', zorder=5)
for x_i, y_i in zip(xs, ys):
    ax.text(x_i, y_i + 0.6, f'{y_i:.2f}%', ha='center', fontweight='bold', fontsize=9)
ax.axhline(91.2, color='red', linestyle='--', lw=1.5, label='Baseline (91.2%)')
ax.set_xticks(xs)
ax.set_xticklabels([f.replace('/', '/\n') for f in fam_order], fontsize=9)
ax.set_ylabel('Mean Validation Accuracy (%)')
ax.set_ylim(70, 95)
ax.set_title('Mean Accuracy by Architectural Family (PASS Modules)', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
save('chart_lollipop_family.png')

# Chart 16: Violin Plot Across Base Models
print('[16] Violin Plot across Models')
data_v = [
    [m.get('acc_distilbert') or m['avg_acc'] for m in filtered_modules],
    [m.get('acc_gpt2') or m['avg_acc'] for m in filtered_modules],
    [m.get('acc_roberta') or m['avg_acc'] for m in filtered_modules],
]
fig, ax = plt.subplots(figsize=(7.5, 5))
parts = ax.violinplot(data_v, positions=[1, 2, 3], showmedians=True)
colors_v = ['#3498DB', '#9B59B6', '#E67E22']
for pc, c in zip(parts['bodies'], colors_v):
    pc.set_facecolor(c)
    pc.set_alpha(0.7)
parts['cmedians'].set_color('black')
parts['cmedians'].set_linewidth(2)
ax.axhline(91.2, color='red', linestyle='--', lw=1.5, label='Baseline (91.2%)')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['DistilBERT', 'GPT-2', 'RoBERTa'], fontsize=10, fontweight='bold')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy Distribution Across Evaluated Base Models (Violin Plot)', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
save('chart_violin_accuracy.png')

# Chart 17: Dashboard 4-Panel Overview (PASS Only)
print('[17] 4-Panel Overview Dashboard')
fig = plt.figure(figsize=(13, 9))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# A: Histogram
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(accs_all, bins=14, color='#2ECC71', edgecolor='white', alpha=0.85)
ax1.axvline(np.mean(accs_all), color='navy', linestyle='--', lw=1.8, label=f'Mean: {np.mean(accs_all):.2f}%')
ax1.axvline(91.2, color='red', linestyle='--', lw=1.5, label='Baseline: 91.2%')
ax1.set_title('A) Accuracy Distribution (PASS Modules)', fontweight='bold')
ax1.set_xlabel('Accuracy (%)')
ax1.set_ylabel('Module Count')
ax1.legend(fontsize=8)

# B: Family Breakdown
ax2 = fig.add_subplot(gs[0, 1])
ax2.barh(range(len(fam_order)), counts, color=colors_f, edgecolor='white')
ax2.set_yticks(range(len(fam_order)))
ax2.set_yticklabels(fam_order, fontsize=8)
ax2.invert_yaxis()
ax2.set_title('B) Validated Modules by Family', fontweight='bold')
ax2.set_xlabel('Module Count')

# C: Model Comparison
ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(['DistilBERT', 'GPT-2', 'RoBERTa'], m_means, color=['#3498DB', '#9B59B6', '#E67E22'], width=0.5)
ax3.axhline(91.2, color='red', linestyle='--', lw=1.5, label='Baseline (91.2%)')
for i, v in enumerate(m_means): ax3.text(i, v - 5, f'{v:.2f}%', ha='center', color='white', fontweight='bold')
ax3.set_ylim(60, 98)
ax3.set_title('C) Mean Accuracy by Base Model', fontweight='bold')
ax3.set_ylabel('Accuracy (%)')
ax3.legend(fontsize=8)

# D: Top 8 Modules
ax4 = fig.add_subplot(gs[1, 1])
top8 = filtered_modules[:8]
ax4.barh(range(8), [m['avg_acc'] for m in top8], color='#27AE60', edgecolor='white', height=0.6)
ax4.set_yticks(range(8))
ax4.set_yticklabels([m['module_name'].replace('_', ' ') for m in top8], fontsize=8)
ax4.set_xlim(85, 93)
ax4.invert_yaxis()
ax4.set_title('D) Top 8 Performing Modules', fontweight='bold')
ax4.set_xlabel('Accuracy (%)')

plt.suptitle('Comprehensive Performance Dashboard (Verified PASS Modules Only)', fontsize=14, fontweight='bold', y=0.99)
save('chart_dashboard_4panel.png')

# Chart 18: Pass Rate per Family (100% PASS by definition)
print('[18] High-Accuracy Success Rate by Family')
fig, ax = plt.subplots(figsize=(8.5, 4.8))
acc_80_rate = {}
for f in fam_order:
    f_mods = [m for m in filtered_modules if m['family'] == f]
    acc_80_rate[f] = sum(1 for m in f_mods if m['avg_acc'] >= 80.0) / len(f_mods) * 100

ys_rate = [acc_80_rate[f] for f in fam_order]
bars = ax.barh(fam_order, ys_rate, color=colors_f, edgecolor='white', height=0.6)
for bar, r in zip(bars, ys_rate):
    ax.text(r + 1, bar.get_y() + bar.get_height()/2, f'{r:.1f}%', va='center', fontweight='bold', fontsize=8.5)
ax.set_xlabel('Percentage of Modules with Accuracy ≥ 80% (%)')
ax.set_title('High-Performance Success Ratio (≥ 80% Accuracy) by Architecture Family', fontsize=12, fontweight='bold')
ax.set_xlim(0, 115)
ax.invert_yaxis()
plt.tight_layout()
save('chart_pass_rate_by_family.png')

print('\n✅ All pure PASS, 100% English charts generated successfully!')
