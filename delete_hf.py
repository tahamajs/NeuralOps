from huggingface_hub import HfApi
import sys

api = HfApi()
modules = [
    'a3', 'adasplash_attention', 'aegis_v5_fn', 'ast_pruner', 'come_factory', 'cure_factory',
    'dynamic_rank_compressor', 'expert_choice_moe', 'flat_llm_factory', 'granite_hybrid',
    'inject_aegis_v2', 'inject_kda', 'inject_minimal_nova', 'inject_vspec_attn', 'inject_weighted_ensemble',
    'kvtc_quantizer', 'lola_attn', 'mamba_module_inject', 'minimal_svd_v2', 'moba_attention',
    'oha', 'oha_inject', 'orion_inject', 'phantom', 'proxsparse', 'regla_attention',
    'replace_attention_inject', 'soa_fused_module', 'svd_llm', 'unified_rope_inject', 'uniql_factory'
]

deleted = []
for m in modules:
    repo_id = f"Hooshaai/svd-linear-attention-gpt2-{m.replace('_', '-')}"
    try:
        api.delete_repo(repo_id=repo_id, repo_type="model")
        deleted.append(repo_id)
        print(f"Deleted {repo_id}")
    except Exception as e:
        print(f"Failed {repo_id}: {e}")

print(f"Deleted {len(deleted)} repos total.")
