import csv, glob
rows = []
for f in sorted(glob.glob('papers/modules/*/runs/rank_ablation/*/*/distilbert/summary.csv')):
    with open(f) as fh:
        r = list(csv.DictReader(fh))
        if r:
            row = r[-1]
            row['path'] = f
            rows.append(row)

# Print corrected table
print(f"{'Module':<30} {'Rank':>5} {'Acc%':>6} {'Comp_Layers':>12} {'Params_After':>14} {'Status':<25}")
print('-' * 100)
for r in rows:
    print(f"{r.get('module',''):<30} {r.get('rank',''):>5} {r.get('accuracy_pct',''):>6} {r.get('num_compressed_layers','0'):>12} {r.get('params_after',''):>14} {r.get('status',''):<25}")
