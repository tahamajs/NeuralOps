import json
import glob
import os
from pathlib import Path
import pandas as pd

lora_mods = [m.strip() for m in open('results/lora_only_modules.txt').read().split(',') if m.strip()]
files = sorted(glob.glob('logs/isolated_results/*.json'), key=os.path.getmtime)

records = []
for f in files:
    try:
        content = open(f, encoding='utf-8').read().strip()
        if content:
            data = json.loads(content)
            if 'module' in data and 'dataset' in data:
                records.append(data)
    except Exception:
        pass

datasets = ['mnli', 'ag_news', 'sst2']
summary_data = {}

for ds in datasets:
    latest = {}
    for r in records:
        if r.get('dataset') == ds and r.get('module') in lora_mods:
            latest[r['module']] = r
    summary_data[ds] = latest

for ds, mods_dict in summary_data.items():
    df = pd.DataFrame(list(mods_dict.values()))
    pass_cnt = (df['status'] == 'PASS').sum()
    pass_low_cnt = (df['status'] == 'PASS_LOW').sum()
    mod_exp_cnt = (df['status'] == 'MODERATE_EXPANSION').sum()
    ext_exp_cnt = (df['status'] == 'EXTREME_EXPANSION').sum()
    crash_cnt = (df['status'] == 'CRASH').sum()
    print(f"Dataset {ds}: total={len(df)}, PASS={pass_cnt}, PASS_LOW={pass_low_cnt}, MOD_EXP={mod_exp_cnt}, EXT_EXP={ext_exp_cnt}, CRASH={crash_cnt}")

    out_csv = Path(f"results/rank_ablation/lora0_{ds}")
    out_csv.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv / "summary.csv", index=False)
    print(f"Saved {out_csv / 'summary.csv'}")
