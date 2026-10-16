import torch
import torch.nn as nn
import copy
from pathlib import Path
from neuralops.compression_engine.svd.direct_svd import DirectSVDCompressor
from scripts.benchmarks.batch_validate_all_compressors import load_roberta, get_probe_data, evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
torch.manual_seed(seed)

print('=== Fair Ablation: Isolating Cholesky Whitening vs Nullspace Projection ===')
print('Verifying identical initialization, seed, calibration data across all conditions.')

# Load base model & data
anchor = load_roberta(device)
probe, heldout, heldout_labels, _ = get_probe_data('roberta', device)
probe_formatted = [{'input_ids': probe[0][:32], 'attention_mask': probe[1][:32]}]

with torch.no_grad():
    y_anchor = anchor(heldout[0], attention_mask=heldout[1]).logits

ranks = [8, 16, 32, 64]
ablation_results = {}

for r in ranks:
    print(f'\n--- Evaluating Rank = {r} ---')
    # 1. Standard Direct SVD (No Whitening, No Nullspace)
    m_direct = copy.deepcopy(anchor)
    c_direct = DirectSVDCompressor(
        model=m_direct,
        rank=r,
        min_rank=1,
        protect_critical=True,
        use_hnc_continuation=False,
        fallback_to_original=False,
        max_retries=0,
    )
    c_direct.compress()
    res_direct = evaluate(m_direct, heldout, y_anchor, 'roberta', device)
    
    # 2. Whitened SVD (Cholesky Whitening, NO Nullspace Stage D)
    m_white = copy.deepcopy(anchor)
    c_white = DirectSVDCompressor(
        model=m_white,
        rank=r,
        min_rank=1,
        protect_critical=True,
        use_hnc_continuation=True,
        probe_inputs=probe_formatted,
        hnc_walk_steps=0,
        hnc_stage_d_steps=0,
        fallback_to_original=False,
        max_retries=0,
    )
    c_white.compress()
    res_white = evaluate(m_white, heldout, y_anchor, 'roberta', device)
    
    # 3. Full HNC-SVD (Cholesky Whitening + Stage D Nullspace Recovery)
    m_hnc = copy.deepcopy(anchor)
    c_hnc = DirectSVDCompressor(
        model=m_hnc,
        rank=r,
        min_rank=1,
        protect_critical=True,
        use_hnc_continuation=True,
        probe_inputs=probe_formatted,
        hnc_walk_steps=0,
        hnc_stage_d_steps=20,
        fallback_to_original=False,
        max_retries=0,
    )
    c_hnc.compress()
    res_hnc = evaluate(m_hnc, heldout, y_anchor, 'roberta', device)
    
    ablation_results[r] = {
        'direct_svd': res_direct['rel_err'],
        'whitened_svd': res_white['rel_err'],
        'hnc_svd': res_hnc['rel_err'],
    }
    print(f'Rank {r:2d} | DirectSVD RelErr: {res_direct["rel_err"]:.4f} | WhitenedSVD RelErr: {res_white["rel_err"]:.4f} | Full HNC-SVD RelErr: {res_hnc["rel_err"]:.4f}')

print('\n=== Final Summary Table ===')
print(f'{"Rank":<6} | {"DirectSVD (Baseline)":<20} | {"WhitenedSVD (No Nullspace)":<26} | {"HNC-SVD (Whitening + Nullspace)":<30}')
print('-' * 90)
for r in ranks:
    d = ablation_results[r]
    print(f'{r:<6} | {d["direct_svd"]:^20.4f} | {d["whitened_svd"]:^26.4f} | {d["hnc_svd"]:^30.4f}')
