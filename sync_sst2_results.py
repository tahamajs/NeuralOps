import json, csv, glob
from pathlib import Path

out_dir = Path('results/rank_ablation/sst2_rank32')
sum_path = out_dir / 'summary.csv'
fields = open(sum_path, encoding='utf-8').readline().strip().split(',')
remaining = ['spectral_residual_lora', 'spectron', 'svd_compressed', 'svd_linear', 'svd_linear_decay', 'svd_llm_v2', 'svd_proj', 'swift_svd', 'swift_svd_class', 'vspec']
existing = {r['module_name']: r for r in csv.DictReader(open(sum_path, encoding='utf-8'))}
added = 0

for m in remaining:
    cands = [f for f in glob.glob(f'logs/isolated_results/{m}_*.json')]
    for c in sorted(cands, reverse=True):
        try:
            d = json.load(open(c, encoding='utf-8'))
            if d.get('dataset') == 'sst2':
                row = {f: d.get(f, '') for f in fields}
                row['module_name'] = m
                row['accuracy_pct'] = d.get('accuracy_pct', d.get('accuracy', 0.0))
                row['f1_score'] = d.get('f1_score', d.get('f1', 0.0))
                row['compression_rate'] = d.get('compression_rate', d.get('compression_ratio', 1.0))
                row['total_params_before'] = d.get('total_params_before', d.get('params_before', 0))
                row['total_params_after'] = d.get('total_params_after', d.get('params_after', 0))
                row['timestamp'] = d.get('finished_at', '')
                existing[m] = row
                met_file = out_dir / f'distilbert_sst2_{m}_full_metrics.json'
                met_file.write_text(json.dumps(d, indent=2), encoding='utf-8')
                added += 1
                break
        except Exception:
            pass

print(f"Added {added} modules. Total now in summary: {len(existing)}")
with open(sum_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader()
    for m in existing:
        w.writerow(existing[m])
