import pandas as pd
import json
import glob
from pathlib import Path

# 1. Zero-shot summary
zs_rows = []
for d in Path('results/zeroshot_parallel').iterdir():
    if not d.is_dir():
        continue
    row = {'module': d.name}
    for t in ['piqa','arc_easy','arc_challenge','hellaswag','winogrande']:
        fp = d / f'{t}.json'
        if fp.exists():
            row[t] = json.load(open(fp)).get('accuracy')
    vals = [row[t] for t in ['piqa','arc_easy','arc_challenge','hellaswag','winogrande'] if t in row and row[t] is not None]
    row['avg'] = round(sum(vals)/len(vals), 2) if vals else None
    zs_rows.append(row)

if zs_rows:
    df = pd.DataFrame(zs_rows)
    baseline = df[df['module'] == 'none']
    if len(baseline):
        b = baseline.iloc[0]['avg']
        df['retention_pct'] = (df['avg'] / b * 100).round(2)
    df.to_csv('audit/zeroshot_summary.csv', index=False)
    print(f'Zero-shot: {len(df)} modules')

# 2. MNLI rank sweep
mnli_rows = []
for f in glob.glob('results/rank_sweep_mnli/*/rank_curve.csv'):
    module = Path(f).parent.name
    df = pd.read_csv(f)
    for _, r in df.iterrows():
        mnli_rows.append({'module': module, **r.to_dict()})
if mnli_rows:
    pd.DataFrame(mnli_rows).to_csv('audit/rank_sweep_mnli_summary.csv', index=False)
    print(f'MNLI: {len(mnli_rows)} rows')

# 3. AG News rank sweep
ag_rows = []
for f in glob.glob('results/rank_sweep_agnews/*/rank_curve.csv'):
    module = Path(f).parent.name
    df = pd.read_csv(f)
    for _, r in df.iterrows():
        ag_rows.append({'module': module, **r.to_dict()})
if ag_rows:
    pd.DataFrame(ag_rows).to_csv('audit/rank_sweep_agnews_summary.csv', index=False)
    print(f'AG News: {len(ag_rows)} rows')

# 4. C4 PPL
c4_rows = []
for f in glob.glob('results/c4_baselines/*/summary.csv'):
    module = Path(f).parent.name
    df = pd.read_csv(f)
    for _, r in df.iterrows():
        c4_rows.append({'module': module, **r.to_dict()})
if c4_rows:
    pd.DataFrame(c4_rows).to_csv('audit/c4_all_summary.csv', index=False)
    print(f'C4: {len(c4_rows)} rows')
