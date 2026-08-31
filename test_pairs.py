import sys, torch  
sys.path.insert(0, 'Codes')  
from framework.run_quality_sweep import evaluate_single_seed_run, _import_all_modules  
_import_all_modules()  
mods = ['adaptive_span', 'adaptive_sparse', 'compressed_attention', 'compressed_attention_layer_energy', 'direct_svd', 'entropy_budget', 'dynamic_rank', 'entropy_rank_allocator', 'flash3', 'flash3_fp8', 'eat', 'fusion_attn', 'halo_regla', 'aegis_hybrid', 'aegis_v2']  
res = {}  
for m in mods:  
    r = evaluate_single_seed_run(module_name=m, model_family='distilbert', dataset='sst2', seed=42, epochs=1, batch_size=64, device=torch.device('cuda:0'), pretrained=True)  
    res[m] = {'acc': r.accuracy_pct, 'loss': r.loss_trajectory[:10]}  
    print(f'FINISHED {m}: acc={r.accuracy_pct} loss={r.loss_trajectory[:3]}')  
import json; json.dump(res, open('D:/SVD_Project/test_pairs_out.json', 'w'), indent=2)  
