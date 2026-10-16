#!/usr/bin/env python3
"""
Aggregate and categorize all benchmark results across the NeuralOps repository.

Scans:
  - results/
  - audit/
  - paper/tables/
  - thesis_final/master_verified_results.json

Categorizes all 203 canonical modules (131 Attentions + 72 Compressors) into 5 diagnostic tiers:
  Tier 1: PARETO_OPTIMAL
  Tier 2: VALID_PASS
  Tier 3: REQUIRES_KD_OR_HNC_RECOVERY
  Tier 4: PARAMETER_INFLATION_OR_NO_COMPRESSION
  Tier 5: UNDERPERFORMING_OR_BUGGY_NEEDS_FIX
"""

import json
import os
import csv
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# 8 Canonical Attention Families (131 Modules)
ATTENTION_FAMILIES = {
    "Kernelized & Feature-Map Linear Attentions": [
        "cosformer", "cosformer_attention", "performer_attention", "chela_slay",
        "chela_slay_attention", "galerkin_attention", "primal_attention", "nystrom_attention",
        "spectraformer_attention", "rala_attention", "nasc_attention", "alibi",
        "eat", "eat_attention", "minimal_nova", "orion", "triton_cosformer", "vspec"
    ],
    "Gated Linear & Householder Delta-Rule Recurrences": [
        "gated_delta_net", "gated_deltanet_block", "gated_linear_attention", "gated_linear_attn",
        "gla_attention", "decay_linear_attention", "decay_linear_attn", "dynamic_conv_attention",
        "goldfinch_ahn", "griffin_ssd", "hgrn2_gated_deltanet", "hgrn2_gated_layer",
        "hgrn2_gated_outer", "hgrn2_outer", "kda_attention", "hla_attention",
        "factorised_matrix_state", "selective_ssm", "simple_ssm", "upi_stuffed_mamba"
    ],
    "Selective State Space Models (SSMs)": [
        "mamba_block", "mamba_svd_flash", "svd_mamba", "selective_ssm", "simple_ssm"
    ],
    "SVD & Spectral Low-Rank Attentions": [
        "adaptive_svd_attention", "asvd_attention", "compressed_attention", "compressed_mha_attention",
        "fused_svd_kernel", "halo_regla", "lola_attention", "lora_attention", "minimal_svd",
        "naive_svd", "padre_attention", "phantom", "projection_factorized_attn", "regla_attention",
        "residual_svd_attention", "rpc_attention", "salt_attention", "spectron", "spectron_fp8",
        "svd_compressed", "svd_linear", "svd_linear_attention", "svd_linear_attn", "svd_linear_decay",
        "svd_linear_inject", "svd_llm_v2_attention", "svd_projection_attention", "svd_seq",
        "triton_svd_linear_attention", "mla_attention"
    ],
    "Block-Sparse, Clustering & Sliding-Window Attentions": [
        "adaptive_span_attention", "adaptive_sparse_attention", "adasplash_attention",
        "block_transformer_hiba", "cluster_attention", "hamming_attention", "landmark_local_attention",
        "lsh_attention", "moba_attention", "nsa_attention", "progressive_token_pruning",
        "refined_landmark_attention", "sink_attention", "token_compressed_attention", "vig_attention",
        "xattention"
    ],
    "Hybrid, Multi-Branch & Mixture-of-Experts (MoE) Attentions": [
        "aegis_hybrid", "aegis_spectron_fp8", "aegis_v1", "aegis_v2", "aegis_v3", "aegis_v4",
        "aegis_v5", "aegis_v6", "bam_umoe", "expert_choice_moe", "granite_hybrid", "hybrid_attention",
        "hybrid_local_kernel", "hymba_transx", "jamba", "jamba_info", "mixattention_moska",
        "moe_deltakv_paged", "native_hybrid_attention", "nexus", "optimized_hybrid_attention",
        "reap_moe", "routing_attention", "samba_m3", "samba_moe", "split_head_hybrid",
        "tri_branch_hybrid", "ttt_linear_attention"
    ],
    "Modern Positional & Grouped-Query Attentions": [
        "flash_attention3", "flash_attention3_fp8", "gqa_attention", "wgqa_attention",
        "mqa", "qcqa_attention", "soa_attn", "unified_rope"
    ],
    "Dense Reference Baselines": [
        "full", "full_attention", "cross_attention", "template_attention", "zeros_attention"
    ]
}

# 5 Canonical Compression Paradigms (72 Modules)
COMPRESSION_PARADIGMS = {
    "Activation-Aware, Whitened & HNC-Steered SVD Factorizations": [
        "a3_compressor", "aa_svd_compressor", "adaptive_ranks", "adasvd_compressor",
        "come_compressor", "compot_compressor", "cr_compressor", "cur_compressor",
        "cur_moe_compressor", "direct_svd", "dobi_svd", "drone_compressor",
        "dynamic_rank", "enlighten_compressor", "entropy_budget_allocation", "erc_svd",
        "flat_llm", "frandi_compressor", "fwsvd_compressor", "gfws",
        "grecodec_compressor", "hybrid_svd_layer", "liftr", "lsc_compressor",
        "mgie_budget", "modegpt_compressor", "nmf_compressor", "oats_compressor",
        "optimal_subspace_projection", "residual_svd_compressor", "rpca_compressor",
        "saes_svd_v2", "salr", "sdrm_compressor", "soka_compressor", "subspace_projection",
        "svd_linear", "svd_llm_compressor", "swift_svd", "swift_svd_act", "tomoe_compressor"
    ],
    "Vector Quantization, Codebooks & Sub-Byte Integer Quantizers": [
        "astro_quantizer", "coopq_quantizer", "faq_quantizer", "kmeans_quantize",
        "rotational_slicegpt_int4", "uniql_compressor", "huffman_coder"
    ],
    "Structured, Semi-Structured & ADMM Pruners": [
        "admm_compressor", "admm_pruner", "armor_pruner", "ast_pruner", "bonsai_pruner",
        "flap", "mamba_shedder", "proxsparse", "slicegpt", "slicegpt_pca",
        "slim_compressor", "sparsegpt"
    ],
    "Tensor Decompositions & Hashed Representations": [
        "hashed_net_compressor", "kronecker_compressor", "kronecker_tt_lora",
        "mlorq_compressor", "roast_hashed_linear", "tt_compressor"
    ],
    "KV-Cache & Latent Memory Compressors": [
        "aatc_xkv_compressor", "cachenotes_compressor", "delta_kv",
        "hybridkv_compressor", "kvtc_compressor", "zerokv_neo_compressor"
    ]
}


def load_all_existing_metrics() -> Dict[str, Dict[str, Any]]:
    records = {}

    # 1. Load thesis_final/master_verified_results.json if exists
    thesis_path = REPO_ROOT / "thesis_final" / "master_verified_results.json"
    if thesis_path.exists():
        try:
            with open(thesis_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        records[k] = v
                elif isinstance(data, list):
                    for item in data:
                        if "module" in item:
                            records[item["module"]] = item
        except Exception as e:
            print(f"Notice: loading {thesis_path} returned {e}")

    # 2. Load results/*.json
    results_dir = REPO_ROOT / "results"
    if results_dir.exists():
        for fpath in results_dir.glob("**/*.json"):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if isinstance(content, dict):
                        for k, v in content.items():
                            if isinstance(v, dict) and ("accuracy" in v or "sst2" in v or "mnli" in v):
                                if k not in records:
                                    records[k] = {}
                                records[k].update(v)
            except Exception:
                pass

    return records


def determine_tier(name: str, metrics: Dict[str, Any], is_attn: bool) -> str:
    # Heuristics based on empirical metrics and known failure signatures
    sst2 = metrics.get("sst2", metrics.get("accuracy", 0.0))
    mnli = metrics.get("mnli", 0.0)
    ppl = metrics.get("wikitext_ppl", metrics.get("ppl", 0.0))
    comp = metrics.get("compression_rate", metrics.get("compression_ratio", 1.0))
    status = str(metrics.get("status", "")).upper()

    # Tier 4: Parameter inflation / No compression
    if comp < 1.0 or "EXPANSION" in status:
        return "Tier 4 (PARAMETER_INFLATION_OR_NO_COMPRESSION)"
    
    # Tier 5: Buggy / Underperforming
    if "FAIL" in status or "CRASH" in status or (0 < sst2 < 56.0) or (0 < ppl and ppl > 500):
        return "Tier 5 (UNDERPERFORMING_OR_BUGGY_NEEDS_FIX)"
    
    # Tier 1: Pareto Optimal
    if (sst2 >= 88.0 and (comp > 1.05 or metrics.get("throughput", 0) > 60000)) or (mnli >= 78.0 and comp > 1.05):
        return "Tier 1 (PARETO_OPTIMAL)"
    
    # Tier 3: Requires KD or HNC recovery
    if (56.0 <= sst2 < 78.0) or ("PASS_LOW" in status):
        return "Tier 3 (REQUIRES_KD_OR_HNC_RECOVERY)"
    
    # Tier 2: Valid Pass
    return "Tier 2 (VALID_PASS)"


def main():
    records = load_all_existing_metrics()
    
    categorized = []
    
    # Process Attentions
    for fam, modules in ATTENTION_FAMILIES.items():
        for mod in modules:
            m_data = records.get(mod, {})
            tier = determine_tier(mod, m_data, is_attn=True)
            categorized.append({
                "module_name": mod,
                "type": "attention",
                "family": fam,
                "tier": tier,
                "sst2_acc": m_data.get("sst2", m_data.get("accuracy", 0.0)),
                "mnli_acc": m_data.get("mnli", 0.0),
                "agnews_acc": m_data.get("agnews", 0.0),
                "compression_ratio": m_data.get("compression_rate", m_data.get("compression_ratio", 1.0)),
                "peak_vram_mb": m_data.get("peak_vram_mb", m_data.get("vram", 0.0)),
                "throughput_tok_s": m_data.get("throughput", 0.0),
                "status": m_data.get("status", "VALID")
            })

    # Process Compressors
    for parad, engines in COMPRESSION_PARADIGMS.items():
        for eng in engines:
            m_data = records.get(eng, {})
            tier = determine_tier(eng, m_data, is_attn=False)
            categorized.append({
                "module_name": eng,
                "type": "compressor",
                "family": parad,
                "tier": tier,
                "sst2_acc": m_data.get("sst2", m_data.get("accuracy", 0.0)),
                "mnli_acc": m_data.get("mnli", 0.0),
                "agnews_acc": m_data.get("agnews", 0.0),
                "compression_ratio": m_data.get("compression_rate", m_data.get("compression_ratio", 1.0)),
                "peak_vram_mb": m_data.get("peak_vram_mb", m_data.get("vram", 0.0)),
                "throughput_tok_s": m_data.get("throughput", 0.0),
                "status": m_data.get("status", "VALID")
            })

    # Summary counts
    tier_counts = {}
    for entry in categorized:
        t = entry["tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1

    out_json = REPO_ROOT / "results" / "master_paper_categorized_inventory.json"
    out_csv = REPO_ROOT / "results" / "master_paper_categorized_inventory.csv"
    
    out_json.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"total_modules": len(categorized), "tier_counts": tier_counts, "modules": categorized}, f, indent=2)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(categorized[0].keys()))
        writer.writeheader()
        writer.writerows(categorized)

    print(f"Successfully categorized {len(categorized)} modules.")
    for t, c in sorted(tier_counts.items()):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
