import os, glob
from collections import defaultdict

# Available 11-section modules
all_files = sorted(glob.glob('thesis_final/attentions_latex/individual_modules/*.tex'))
files_11 = []
for f in all_files:
    with open(f, 'r', encoding='utf-8') as fl:
        c = fl.read()
    if 'بخش ۱:' in c and 'بخش ۱۱:' in c:
        files_11.append(os.path.splitext(os.path.basename(f))[0])

def stem(name):
    n = name.lower()
    for s in ['_attention', '_module', '_attn', '_layer', '_block', '_compressor']:
        if n.endswith(s):
            n = n[:-len(s)]
    return n

# Current modules in chapter_attentions.tex
with open('thesis_final/chapter_attentions.tex') as f:
    curr = [os.path.splitext(os.path.basename(l.strip().replace(r'\input{', '').rstrip('}')))[0] for l in f if 'input{' in l]

print(f"Current count in chapter_attentions: {len(curr)}")

# Priority list: Keep existing unique modules from curr, discard duplicates
seen_stems = set()
selected_100 = []

# Specific preferences for duplicates in curr:
canonical_picks = {
    'spectra_v2': 'spectra_v2_attention',
    'moba': 'moba_attention',
    'spectral_mamba': 'spectral_mamba_attention',
    'eat': 'eat_attention',
    'cluster': 'cluster_attention',
    'wgqa': 'wgqa_attention',
    'rpc': 'rpc_attention',
    'svd_llm_v2': 'svd_llm_v2_attention',
    'aegis_hybrid': 'aegis_hybrid_attention',
    'svd_compressed': 'svd_compressed_attention',
    'spectron': 'spectron_attention',
    'compressed_mha': 'compressed_mha_attention',
    'soa': 'soa_fused_module',
    'squeezing_heads': 'squeezing_heads_distiller',
    'svd_mamba': 'svd_mamba',
    'nha': 'nha_attention',
    'minimal_svd': 'minimal_svd_v2',
}

# Add distinct modules from curr
for m in curr:
    if m == 'SVD-Mamba':
        m = 'svd_mamba'
    s = stem(m)
    # Check if this stem has a canonical pick
    if s in canonical_picks:
        pick = canonical_picks[s]
        if pick not in selected_100 and stem(pick) not in seen_stems:
            selected_100.append(pick)
            seen_stems.add(stem(pick))
    elif s not in seen_stems:
        # Also filter out aegis versions (keep only aegis_v1, aegis_spectron_fp8, aegis_hybrid_attention)
        if s in ['aegis_v2', 'aegis_v3', 'aegis_v4', 'aegis_v5']:
            continue
        # Also filter out minimal variants (keep minimal_svd_v2, minimal_aurora)
        if s in ['minimal_nova', 'minimal_prism']:
            continue
        selected_100.append(m)
        seen_stems.add(s)

print(f"Unique kept from curr: {len(selected_100)}")

# Fill up to 100 with distinct, high-impact modules from files_11
candidate_pool = [
    'linear_attention',
    'reformer_attention',
    'swin_attention',
    'sinkhorn_attention',
    'sliding_window_attention',
    'ring_attention',
    'fourier_attention',
    'dilated_attention',
    'block_sparse_attention',
    'dynamic_entropy_attention',
    'dynamic_sparse_attention',
    'efficient_attention',
    'hyper_attention',
    'local_attention',
    'generalized_attention',
    'factorized_attention',
    'vector_quantized_attention',
    'tree_attention',
    'syntactic_attention',
    'wave_attention',
    'gated_attention',
    'cosformer',
    'retention'
]

for cand in candidate_pool:
    if len(selected_100) >= 100:
        break
    s = stem(cand)
    if s not in seen_stems and cand in files_11:
        selected_100.append(cand)
        seen_stems.add(s)

# If still < 100, add any other unique stem from files_11
if len(selected_100) < 100:
    for f in files_11:
        s = stem(f)
        if s not in seen_stems:
            selected_100.append(f)
            seen_stems.add(s)
            if len(selected_100) == 100:
                break

print(f"Final selected unique modules: {len(selected_100)}")

# Verify NO duplicate stems
final_stems = [stem(m) for m in selected_100]
assert len(final_stems) == len(set(final_stems)), "Duplicate stems found!"
print("Verification SUCCESS: All 100 modules have 100% unique stems!")

# Print the 100 modules
for idx, m in enumerate(selected_100, 1):
    print(f"{idx:2d}: {m}")

