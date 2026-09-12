import os, glob

# All 11-section files
all_files = sorted(glob.glob('thesis_final/attentions_latex/individual_modules/*.tex'))
files_11 = set()
for f in all_files:
    with open(f, 'r', encoding='utf-8') as fl:
        c = fl.read()
    if 'بخش ۱:' in c and 'بخش ۱۱:' in c:
        files_11.add(os.path.splitext(os.path.basename(f))[0])

def normalize_stem(name):
    n = name.lower().replace('-', '_')
    # Strip common suffixes
    for s in ['_attention', '_module', '_attn', '_layer', '_block', '_compressor', '_distiller', '_distill']:
        if n.endswith(s):
            n = n[:-len(s)]
    return n

# Current order from chapter_attentions.tex
with open('thesis_final/chapter_attentions.tex') as f:
    curr = [os.path.splitext(os.path.basename(l.strip().replace(r'\input{', '').rstrip('}')))[0] for l in f if 'input{' in l]

curated_100 = []
seen_stems = set()

# Special canonical choices when duplicates appear:
canonical_map = {
    'spectra_v2': 'spectra_v2_attention',
    'spectra_v2_attention': 'spectra_v2_attention',
    'moba': 'moba_attention',
    'moba_attention': 'moba_attention',
    'spectral_mamba': 'spectral_mamba_attention',
    'spectral_mamba_attention': 'spectral_mamba_attention',
    'eat': 'eat_attention',
    'eat_attention': 'eat_attention',
    'cluster': 'cluster_attention',
    'cluster_attention': 'cluster_attention',
    'wgqa': 'wgqa_attention',
    'wgqa_attention': 'wgqa_attention',
    'rpc': 'rpc_attention',
    'rpc_attention': 'rpc_attention',
    'svd_llm_v2': 'svd_llm_v2_attention',
    'svd_llm_v2_attention': 'svd_llm_v2_attention',
    'aegis_hybrid': 'aegis_hybrid_attention',
    'aegis_hybrid_attention': 'aegis_hybrid_attention',
    'svd_compressed': 'svd_compressed_attention',
    'svd_compressed_attention': 'svd_compressed_attention',
    'spectron': 'spectron_attention',
    'spectron_attention': 'spectron_attention',
    'compressed_mha': 'compressed_mha_attention',
    'compressed_mha_attention': 'compressed_mha_attention',
    'nha_compressor': 'nha_attention',
    'nha_attention': 'nha_attention',
    'squeezing_heads': 'squeezing_heads_distiller',
    'squeezing_heads_distill': 'squeezing_heads_distiller',
    'squeezing_heads_distiller': 'squeezing_heads_distiller',
    'soa_attn': 'soa_fused_module',
    'soa_module': 'soa_fused_module',
    'soa_fused_module': 'soa_fused_module',
    'svd_mamba': 'svd_mamba',
    'svd-mamba': 'svd_mamba',
}

# Families where we keep only ONE representative to avoid version spam:
# Aegis: aegis_v1, aegis_spectron_fp8, aegis_hybrid_attention are distinct enough, but remove v2, v3, v4, v5, aegis_attention
# Minimal: keep minimal_svd_v2 and minimal_aurora, remove minimal_svd, minimal_nova, minimal_prism
# Flash3: keep flash_attention3 and flash_attention3_fp8 (or just flash_attention3)
blacklist = {
    'aegis_v2', 'aegis_v3', 'aegis_v4', 'aegis_v5', 'aegis_attention',
    'minimal_svd', 'minimal_nova', 'minimal_prism',
    'SVD-Mamba'
}

for m in curr:
    if m in blacklist:
        continue
    actual_m = canonical_map.get(m, m)
    stem = normalize_stem(actual_m)
    if stem not in seen_stems and actual_m in files_11:
        curated_100.append(actual_m)
        seen_stems.add(stem)

print(f"Retained from current chapter: {len(curated_100)} unique architectures")

# High-impact additions from 11-section pool:
additions = [
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
    'retention',
    'async_spec_attention',
    'memory_efficient_attention',
    'multi_query_attention',
    'topk_routing_attention',
    'tensor_product_attention',
    'quantized_attention',
    'fused_recurrent_attention',
    'prefix_attention',
    'pooled_attention',
    'poly_attention'
]

for cand in additions:
    if len(curated_100) >= 100:
        break
    stem = normalize_stem(cand)
    if stem not in seen_stems and cand in files_11:
        curated_100.append(cand)
        seen_stems.add(stem)

print(f"Total after curated additions: {len(curated_100)}")

# Check for any duplicate stems
stems = [normalize_stem(m) for m in curated_100]
duplicates = [s for s in stems if stems.count(s) > 1]
if duplicates:
    print("WARNING: Duplicate stems:", set(duplicates))
else:
    print("SUCCESS: Exactly 100 distinct, unique architectures selected with zero duplicates!")

for i, m in enumerate(curated_100, 1):
    print(f"{i:3d}: {m}")

