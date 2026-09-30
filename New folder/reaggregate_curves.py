import glob
import os
import pandas as pd

base = r'D:\SVD_Project\results\04_rank_sweeps\distilbert\deep_rank_distilbert'
all_curves = []
for mod in sorted(os.listdir(base)):
    mod_dir = os.path.join(base, mod)
    if not os.path.isdir(mod_dir):
        continue
    rows = []
    for r_dir in sorted(os.listdir(mod_dir)):
        sum_f = os.path.join(mod_dir, r_dir, 'summary.csv')
        if os.path.isfile(sum_f) and os.path.getsize(sum_f) > 0:
            try:
                df = pd.read_csv(sum_f)
                if not df.empty:
                    r_val = r_dir.replace('rank_', '')
                    df['rank'] = r_val
                    rows.append(df.iloc[0])
            except Exception:
                pass
    if rows:
        mod_df = pd.DataFrame(rows)
        mod_df.to_csv(os.path.join(mod_dir, 'rank_curve.csv'), index=False)
        all_curves.append(mod_df)

if all_curves:
    pd.concat(all_curves, ignore_index=True).to_csv(os.path.join(base, 'all_rank_curves.csv'), index=False)
print(f"Re-aggregated all rank curves for {len(all_curves)} modules.")
