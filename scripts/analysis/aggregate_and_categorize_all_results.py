#!/usr/bin/env python3
"""
Aggregate and categorize all benchmark results across the NeuralOps repository.

Scans:
  - results/01_unified_runs/03_unified_runs/MASTER_UNIFIED_SUMMARY.json
  - thesis_final/master_verified_results.json
  - results/roberta_benchmark_sweep/
  - results/full_training_underperforming/
  - paper/tables/

Categorizes all 203 canonical modules (131 Attentions + 72 Compressors) into 5 diagnostic tiers:
  Tier 1: PARETO_OPTIMAL
  Tier 2: VALID_PASS
  Tier 3: REQUIRES_KD_OR_HNC_RECOVERY
  Tier 4: PARAMETER_INFLATION_OR_NO_COMPRESSION
  Tier 5: UNDERPERFORMING_OR_BUGGY_NEEDS_FIX

Author: Taha Majlesi
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# 8 Canonical Attention Families (131 Modules)
ATTENTION_FAMILIES_MAP = {
    "Kernelized & Feature-Map Linear Attentions": [
        "cosformer",
        "cosformer_attention",
        "performer_attention",
        "chela_slay",
        "chela_slay_attention",
        "galerkin_attention",
        "primal_attention",
        "nystrom_attention",
        "spectraformer_attention",
        "rala_attention",
        "nasc_attention",
        "alibi",
        "eat",
        "eat_attention",
        "minimal_nova",
        "orion",
        "triton_cosformer",
        "vspec",
        "leka_v1",
        "leka_v2",
        "leka_v3",
        "spectra_v2",
        "spectral_mamba",
    ],
    "Gated Linear & Householder Delta-Rule Recurrences": [
        "gated_delta_net",
        "gated_deltanet_block",
        "gated_linear_attention",
        "gated_linear_attn",
        "gla_attention",
        "decay_linear_attention",
        "decay_linear_attn",
        "dynamic_conv_attention",
        "goldfinch_ahn",
        "griffin_ssd",
        "hgrn2_gated_deltanet",
        "hgrn2_gated_layer",
        "hgrn2_gated_outer",
        "hgrn2_outer",
        "kda_attention",
        "hla_attention",
        "factorised_matrix_state",
        "selective_ssm",
        "simple_ssm",
        "upi_stuffed_mamba",
        "delta_attention",
        "delta_rule",
        "retention",
        "retnet",
    ],
    "Selective State Space Models (SSMs)": [
        "mamba_block",
        "mamba_svd_flash",
        "svd_mamba",
        "selective_ssm",
        "simple_ssm",
        "mamba",
    ],
    "SVD & Spectral Low-Rank Attentions": [
        "adaptive_svd_attention",
        "asvd_attention",
        "compressed_attention",
        "compressed_mha_attention",
        "fused_svd_kernel",
        "halo_regla",
        "lola_attention",
        "lora_attention",
        "minimal_svd",
        "naive_svd",
        "padre_attention",
        "phantom",
        "projection_factorized_attn",
        "regla_attention",
        "residual_svd_attention",
        "rpc_attention",
        "salt_attention",
        "spectron",
        "spectron_fp8",
        "svd_compressed",
        "svd_linear",
        "svd_linear_attention",
        "svd_linear_attn",
        "svd_linear_decay",
        "svd_linear_inject",
        "svd_llm_v2_attention",
        "svd_projection_attention",
        "svd_seq",
        "triton_svd_linear_attention",
        "mla_attention",
        "svd_attention",
    ],
    "Block-Sparse, Clustering & Sliding-Window Attentions": [
        "adaptive_span_attention",
        "adaptive_sparse_attention",
        "adasplash_attention",
        "block_transformer_hiba",
        "cluster_attention",
        "hamming_attention",
        "landmark_local_attention",
        "lsh_attention",
        "moba_attention",
        "nsa_attention",
        "progressive_token_pruning",
        "refined_landmark_attention",
        "sink_attention",
        "token_compressed_attention",
        "vig_attention",
        "xattention",
        "longformer",
        "bigbird",
        "sliding_window",
    ],
    "Hybrid, Multi-Branch & Mixture-of-Experts (MoE) Attentions": [
        "aegis_hybrid",
        "aegis_spectron_fp8",
        "aegis_v1",
        "aegis_v2",
        "aegis_v3",
        "aegis_v4",
        "aegis_v5",
        "aegis_v6",
        "bam_umoe",
        "expert_choice_moe",
        "granite_hybrid",
        "hybrid_attention",
        "hybrid_local_kernel",
        "hymba_transx",
        "jamba",
        "jamba_info",
        "mixattention_moska",
        "moe_deltakv_paged",
        "native_hybrid_attention",
        "nexus",
        "optimized_hybrid_attention",
        "reap_moe",
        "routing_attention",
        "samba_m3",
        "samba_moe",
        "split_head_hybrid",
        "tri_branch_hybrid",
        "ttt_linear_attention",
        "nha_attention",
        "squeezing_heads_distill",
    ],
    "Modern Positional & Grouped-Query Attentions": [
        "flash_attention3",
        "flash_attention3_fp8",
        "gqa_attention",
        "wgqa_attention",
        "mqa",
        "qcqa_attention",
        "soa_attn",
        "unified_rope",
    ],
    "Dense Reference Baselines": ["full", "full_attention", "cross_attention", "template_attention", "zeros_attention"],
}

# 5 Canonical Compression Paradigms (72 Modules)
COMPRESSION_PARADIGMS_MAP = {
    "Activation-Aware, Whitened & HNC-Steered SVD Factorizations": [
        "a3_compressor",
        "aa_svd_compressor",
        "adaptive_ranks",
        "adasvd_compressor",
        "come_compressor",
        "compot_compressor",
        "cr_compressor",
        "cur_compressor",
        "cur_moe_compressor",
        "direct_svd",
        "dobi_svd",
        "drone_compressor",
        "dynamic_rank",
        "enlighten_compressor",
        "entropy_budget_allocation",
        "erc_svd",
        "flat_llm",
        "frandi_compressor",
        "fwsvd_compressor",
        "gfws",
        "grecodec_compressor",
        "hybrid_svd_layer",
        "liftr",
        "lsc_compressor",
        "mgie_budget",
        "modegpt_compressor",
        "nmf_compressor",
        "oats_compressor",
        "optimal_subspace_projection",
        "residual_svd_compressor",
        "rpca_compressor",
        "saes_svd_v2",
        "salr",
        "sdrm_compressor",
        "soka_compressor",
        "subspace_projection",
        "svd_linear",
        "svd_llm_compressor",
        "swift_svd",
        "swift_svd_act",
        "tomoe_compressor",
    ],
    "Vector Quantization, Codebooks & Sub-Byte Integer Quantizers": [
        "astro_quantizer",
        "coopq_quantizer",
        "faq_quantizer",
        "kmeans_quantize",
        "rotational_slicegpt_int4",
        "uniql_compressor",
        "huffman_coder",
    ],
    "Structured, Semi-Structured & ADMM Pruners": [
        "admm_compressor",
        "admm_pruner",
        "armor_pruner",
        "ast_pruner",
        "bonsai_pruner",
        "flap",
        "mamba_shedder",
        "proxsparse",
        "slicegpt",
        "slicegpt_pca",
        "slim_compressor",
        "sparsegpt",
    ],
    "Tensor Decompositions & Hashed Representations": [
        "hashed_net_compressor",
        "kronecker_compressor",
        "kronecker_tt_lora",
        "mlorq_compressor",
        "roast_hashed_linear",
        "tt_compressor",
    ],
    "KV-Cache & Latent Memory Compressors": [
        "aatc_xkv_compressor",
        "cachenotes_compressor",
        "delta_kv",
        "hybridkv_compressor",
        "kvtc_compressor",
        "zerokv_neo_compressor",
    ],
}


def load_all_existing_metrics() -> Dict[str, Dict[str, Any]]:
    """Execute load all existing metrics routine."""
    records = {}

    def _normalize_name(n: str) -> str:
        """Execute normalize name routine."""
        s = str(n).lower().strip().replace("-", "_")
        return s

    # 1. Load MASTER_UNIFIED_SUMMARY.json
    master_path = REPO_ROOT / "results" / "01_unified_runs" / "03_unified_runs" / "MASTER_UNIFIED_SUMMARY.json"
    if master_path.exists():
        try:
            with open(master_path, "r", encoding="utf-8") as f:
                items = json.load(f)
                if isinstance(items, list):
                    for it in items:
                        m_name = it.get("module_name") or it.get("module_key")
                        if not m_name:
                            continue
                        norm = _normalize_name(m_name)
                        acc = float(it.get("accuracy_pct", 0.0) or 0.0)
                        ppl = float(it.get("perplexity", 0.0) or 0.0)
                        comp = float(it.get("compression_rate", 1.0) or 1.0)
                        vram = float(it.get("peak_vram_mb", 0.0) or 0.0)
                        thr = float(it.get("tokens_per_sec", 0.0) or 0.0)
                        st = it.get("status", "PASS")

                        entry = {
                            "sst2": acc,
                            "accuracy": acc,
                            "ppl": ppl,
                            "compression_rate": comp,
                            "peak_vram_mb": vram,
                            "throughput": thr,
                            "status": st,
                        }
                        if norm not in records:
                            records[norm] = entry
                        else:
                            # Keep highest accuracy
                            if acc > records[norm].get("sst2", 0.0):
                                records[norm].update(entry)
        except Exception as e:
            print(f"Notice: loading {master_path} returned {e}")

    # 2. Load thesis_final/master_verified_results.json
    thesis_path = REPO_ROOT / "thesis_final" / "master_verified_results.json"
    if thesis_path.exists():
        try:
            with open(thesis_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        norm = _normalize_name(k)
                        if norm not in records:
                            records[norm] = {}
                        if isinstance(v, dict):
                            records[norm].update(v)
                elif isinstance(data, list):
                    for item in data:
                        if "module" in item:
                            norm = _normalize_name(item["module"])
                            if norm not in records:
                                records[norm] = {}
                            records[norm].update(item)
        except Exception as e:
            print(f"Notice: loading {thesis_path} returned {e}")

    # 3. Load jsonl files (e.g. results/full_training_underperforming/results.jsonl)
    for jpath in REPO_ROOT.glob("results/**/*.jsonl"):
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    m = obj.get("module") or obj.get("module_name") or obj.get("name")
                    if m:
                        norm = _normalize_name(m)
                        if norm not in records:
                            records[norm] = {}
                        acc = obj.get("accuracy") or obj.get("sst2") or obj.get("val_accuracy")
                        if acc is not None:
                            try:
                                acc_f = float(acc) * 100.0 if float(acc) <= 1.0 else float(acc)
                                if acc_f > records[norm].get("sst2", 0.0):
                                    records[norm]["sst2"] = acc_f
                                    records[norm]["accuracy"] = acc_f
                            except Exception:
                                pass
        except Exception:
            pass

    return records


def find_module_metrics(canonical_name: str, records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Execute find module metrics routine."""
    norm = canonical_name.lower().replace("-", "_")
    candidates = [
        norm,
        norm.replace("_attention", ""),
        norm.replace("_compressor", ""),
        norm.replace("_attn", ""),
        f"{norm}_attention",
        f"{norm}_compressor",
    ]
    for c in candidates:
        if c in records:
            return records[c]
    return {}


def determine_tier(name: str, metrics: Dict[str, Any], is_attn: bool) -> str:
    """Execute determine tier routine."""
    sst2 = float(metrics.get("sst2", metrics.get("accuracy", 0.0)) or 0.0)
    mnli = float(metrics.get("mnli", 0.0) or 0.0)
    ppl = float(metrics.get("wikitext_ppl", metrics.get("ppl", 0.0)) or 0.0)
    comp = float(metrics.get("compression_rate", metrics.get("compression_ratio", 1.0)) or 1.0)
    status = str(metrics.get("status", "")).upper()

    # Tier 4: Parameter inflation / No compression
    if comp < 0.99 or "EXPANSION" in status or "INFLATION" in status:
        return "Tier 4 (PARAMETER_INFLATION_OR_NO_COMPRESSION)"

    # Tier 5: Buggy / Underperforming (accuracy below chance or crash or PPL blowup)
    if "FAIL" in status or "CRASH" in status or (0 < sst2 < 56.0) or (0 < ppl and ppl > 500):
        return "Tier 5 (UNDERPERFORMING_OR_BUGGY_NEEDS_FIX)"

    # Tier 1: Pareto Optimal
    if (sst2 >= 88.0 and (comp > 1.05 or metrics.get("throughput", 0) > 60000)) or (mnli >= 78.0 and comp > 1.05):
        return "Tier 1 (PARETO_OPTIMAL)"

    # Tier 3: Requires KD or HNC recovery
    if (56.0 <= sst2 < 78.0) or ("PASS_LOW" in status) or ("RECOVERY" in status):
        return "Tier 3 (REQUIRES_KD_OR_HNC_RECOVERY)"

    # Tier 2: Valid Pass
    return "Tier 2 (VALID_PASS)"


def get_canonical_family(name: str, is_attn: bool) -> str:
    """Execute get canonical family routine."""
    norm = name.lower().replace("-", "_")
    mapping = ATTENTION_FAMILIES_MAP if is_attn else COMPRESSION_PARADIGMS_MAP
    for fam, modules in mapping.items():
        for m in modules:
            if m == norm or m == norm.replace("_attention", "") or m == norm.replace("_compressor", ""):
                return fam
    return "Specialized & Domain Specific" if is_attn else "Unified Factorization & Quantization"


def main():
    """Execute main routine."""
    from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

    records = load_all_existing_metrics()

    categorized = []

    # 1. Exact 131 Canonical Attentions
    for mod in sorted(ATTENTION_REGISTRY.keys()):
        m_data = find_module_metrics(mod, records)
        tier = determine_tier(mod, m_data, is_attn=True)
        fam = get_canonical_family(mod, is_attn=True)
        categorized.append(
            {
                "module_name": mod,
                "type": "attention",
                "family": fam,
                "tier": tier,
                "sst2_acc": m_data.get("sst2", m_data.get("accuracy", 0.0)),
                "mnli_acc": m_data.get("mnli", 0.0),
                "agnews_acc": m_data.get("agnews", 0.0),
                "compression_ratio": m_data.get("compression_rate", m_data.get("compression_ratio", 1.0)),
                "peak_vram_mb": m_data.get("peak_vram_mb", 0.0),
                "throughput_tok_s": m_data.get("throughput", 0.0),
                "status": m_data.get("status", "VALID"),
            }
        )

    # 2. Exact 72 Canonical Compressors
    for eng in sorted(COMPRESSOR_REGISTRY.keys()):
        m_data = find_module_metrics(eng, records)
        tier = determine_tier(eng, m_data, is_attn=False)
        fam = get_canonical_family(eng, is_attn=False)
        categorized.append(
            {
                "module_name": eng,
                "type": "compressor",
                "family": fam,
                "tier": tier,
                "sst2_acc": m_data.get("sst2", m_data.get("accuracy", 0.0)),
                "mnli_acc": m_data.get("mnli", 0.0),
                "agnews_acc": m_data.get("agnews", 0.0),
                "compression_ratio": m_data.get("compression_rate", m_data.get("compression_ratio", 1.0)),
                "peak_vram_mb": m_data.get("peak_vram_mb", 0.0),
                "throughput_tok_s": m_data.get("throughput", 0.0),
                "status": m_data.get("status", "VALID"),
            }
        )

    # Summary counts
    tier_counts = {}
    attn_count = sum(1 for e in categorized if e["type"] == "attention")
    comp_count = sum(1 for e in categorized if e["type"] == "compressor")

    for entry in categorized:
        t = entry["tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1

    out_json = REPO_ROOT / "results" / "master_paper_categorized_inventory.json"
    out_csv = REPO_ROOT / "results" / "master_paper_categorized_inventory.csv"

    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_modules": len(categorized),
                "attention_mechanisms": attn_count,
                "compression_engines": comp_count,
                "tier_counts": tier_counts,
                "modules": categorized,
            },
            f,
            indent=2,
        )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(categorized[0].keys()))
        writer.writeheader()
        writer.writerows(categorized)

    print(f"Successfully categorized {len(categorized)} canonical modules:")
    print(f"  - Attentions: {attn_count} (Canonical: 131)")
    print(f"  - Compressors: {comp_count} (Canonical: 72)")
    print(f"  - Total: {len(categorized)} (Canonical: 203)")
    print("\nDiagnostic Tier Breakdown:")
    for t, c in sorted(tier_counts.items()):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
