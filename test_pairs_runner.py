import sys, os
sys.path.insert(0, r'D:\SVD_Project\Codes')
import torch
from framework.run_quality_sweep import evaluate_single_seed_run, _import_all_modules

_import_all_modules()

mods = [
    'adaptive_span', 'adaptive_sparse',
    'compressed_attention', 'compressed_attention_layer_energy',
    'direct_svd', 'entropy_budget',
    'dynamic_rank', 'entropy_rank_allocator',
    'flash3', 'flash3_fp8',
    'eat', 'fusion_attn',
    'halo_regla',
    'aegis_hybrid', 'aegis_v2'
]

res = {}
for m in mods:
    print(f"--> Starting {m}...")
    try:
        r = evaluate_single_seed_run(
            module_name=m,
            model_family='distilbert',
            dataset='sst2',
            seed=42,
            epochs=1,
            batch_size=64,
            device=torch.device('cuda:0'),
            pretrained=True
        )
        res[m] = {
            'status': r.status,
            'acc': round(r.accuracy_pct, 4),
            'f1': round(r.f1_score, 4),
            'loss_10': [round(x, 4) for x in r.loss_trajectory[:10]]
        }
        print(f"    DONE {m}: acc={r.accuracy_pct:.2f}% loss[0:5]={r.loss_trajectory[:5]}")
    except Exception as e:
        print(f"    FAIL {m}: {e}")
        res[m] = {'status': 'FAIL', 'error': str(e)}

import json
with open(r'D:\SVD_Project	est_pairs_out.json', 'w') as f:
    json.dump(res, f, indent=2)
print("=== ALL FINISHED ===")
