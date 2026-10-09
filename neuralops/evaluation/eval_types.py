"""Types, data structures, and constants for the NeuralOps evaluation engine.

Provides the BenchmarkResult dataclass and configuration maps for model families,
tracks, optimizer classifications, and dataset categories.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

# =============================================================================
# Constants & Model/Dataset Mappings
# =============================================================================

MODEL_TOKENIZER_MAPPING: dict[str, str] = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "t5": "t5-small",
    "gpt2": "gpt2",
    "distilbert": "distilbert-base-uncased",
    "qwen": "Qwen/Qwen2.5-0.5B",
    "qwen2": "Qwen/Qwen2.5-0.5B",
    "qwen3.5": "Qwen/Qwen2.5-0.5B",
    "qwen3_5": "Qwen/Qwen2.5-0.5B",
}

LM_DATASETS: set[str] = {
    "wikitext2",
    "wikitext-2-raw-v1",
    "wikitext",
    "wikitext103",
    "c4",
    "ptb",
    "lm",
}

GLUE_DATASETS: set[str] = {
    "sst2",
    "cola",
    "mrpc",
    "qqp",
    "mnli",
    "qnli",
    "rte",
    "wnli",
    "stsb",
}

ACTIVATION_AWARE_MODULES: set[str] = {
    "a3_compressor",
    "a3",
    "aasvd",
    "asvd",
    "adasvd_compressor",
    "adasvd",
    "flat_llm",
    "armor",
    "armor_pruner",
    "armor_compressor",
    "flap",
    "mgie",
    "mgie_svd",
    "mgie_budget",
    "frandi",
    "frandi_compressor",
    "optimal_subspace_projection",
    "slicegpt",
    "slicegpt_pca",
    "rotational_slicegpt_int4",
    "rotational_slicegpt",
    "proxsparse",
    "mamba_shedder",
    "mlorq",
    "mlorq_compressor",
    "compot",
    "compot_compressor",
    "hashed_net",
    "hashed_net_compressor",
    "lsc",
    "lsc_compressor",
    "grecodec",
    "grecodec_compressor",
    "direct_svd",
    "adaptive_ranks",
    "entropy_budget",
    "entropy_budget_allocation",
    "dynamic_rank",
    "dobi_svd",
    "dobi",
    "erc",
    "erc_svd",
    "gfws",
    "gfws_d_compressor",
    "modegpt",
    "subspace_projection",
    "roast",
    "bonsai",
    "bonsai_pruner",
    "swift_svd",
    "swift_svd_act",
    "hybrid_svd",
}

COMPUTE_OPTIMIZERS: set[str] = {
    "nha",
    "nha_class",
    "nha_attention",
    "block_transformer_hiba",
    "flash3",
    "flash_attention3",
    "flash3_fp8",
    "adasplash",
    "adasplash_attention",
    "moba",
    "xattention",
    "fasa",
    "cluster",
    "cluster_attention",
    "cosformer",
}

KV_CACHE_OPTIMIZERS: set[str] = {
    "hybridkv",
    "hybridkv_class",
    "aatc_xkv",
    "aatc_xkv_compressor",
    "kvtc",
    "delta_kv",
    "delta_kv_class",
    "zerokv",
    "zerokv_neo",
    "moe_deltakv_paged",
    "padre",
}

DEPRECATED_MODULES: set[str] = {
    "eat",
    "eat_attention",
    "granite_hybrid",
    # jamba_info: removed from deprecated list (bugs fixed 2026-10-09)
    "tpa",
    "tpa_attention",
    "tp_attention",
    "tt",
    "tt_linear",
    "tt_compressor",
    "tt_linear_compressor",
    "tt_factory",
    "tt_compressor_factory",
    "tt_linear_factory",
    "lillama_compressor",
    "mamba_svd_flash",
}

TRACK_MODULES: dict[str, dict[str, list[str]]] = {
    "track1_svd": {
        "attentions": [
            "svd_linear_attention",
            "adaptive_svd_attention",
            "asvd_attention",
            "linformer_attention",
            "res_svd",
            "svd_attention",
            "svd_compressed",
            "svd_projection_attention",
            "svd_llm_v2_attention",
            "factorised_matrix_state",
            "mamba_svd_flash",
            "minimal_svd",
            "minimal_svd_v2",
        ],
        "compressors": [
            "adaptive_ranks",
            "direct_svd",
            "dobi_svd",
            "erc_svd",
            "entropy_budget_allocation",
            "kronecker_tt_lora",
            "saes_svd_v2",
            "swift_svd",
            "swift_svd_act",
            "aasvd",
            "rpca",
            "cur_moe",
            "grecodec",
            "frandi",
            "dynamic_rank",
            "hybrid_svd",
            "svd_linear",
            "svd_llm",
            "svd_llm_compressor",
        ],
    },
    "track2_ssm": {
        "attentions": [
            "aegis_v1",
            "aegis_v2",
            "aegis_v3",
            "aegis_v4",
            "aegis_v5",
            "aegis_v6",
            "aegis_hybrid",
            "aegis_spectron_fp8",
            "spectra_v2",
            "spectral_mamba",
            "Mamba",
            "SVD-Mamba",
            "simple_ssm",
            "gated_delta_net",
            "gated_deltanet_block",
            "gated_linear_attention",
            "hgrn2_layer",
            "hgrn2_gated_deltanet",
            "hgrn2_outer",
            "griffin_ssd",
            "regla_attention",
            "halo_regla",
            "kda_attention",
            "decay_linear_attention",
            "spectron",
            "galerkin_attention",
            "vspec",
            "vspec_attention",
        ],
        "compressors": ["mamba_shedder"],
    },
    "track3_moe": {
        "attentions": [
            "bam_umoe",
            "expert_choice_moe",
            "reap_moe",
            "samba_moe",
            "samba_m3",
            "jamba",
            "jamba_info",
            "hymba_transx",
            "moe_deltakv_paged",
            "block_transformer_hiba",
            "nexus",
            "leka_v1",
            "leka_v2",
            "leka_v3",
            "mixattention_moska",
            "phantom",
            "moba",
            "fasa",
            "upi_stuffed_mamba",
            "granite_hybrid",
        ],
        "compressors": ["cur_moe"],
    },
    "track4_kernel_quant": {
        "attentions": [
            "performer_attention",
            "nystrom_attention",
            "flash_attention3",
            "cluster_attention",
            "adasplash_attention",
            "hamming_attention",
            "cosformer",
            "decay_linear_attention",
            "gated_linear_attention",
            "SOAFusedAttention",
            "fused_svd_kernel",
        ],
        "compressors": [
            "astro_quantizer",
            "faq_quantizer",
            "coopq_quantizer",
            "kmeans_quantize",
            "zerokv_neo",
            "modegpt",
            "delta_kv",
            "kvtc",
            "aatc_xkv",
            "hybridkv",
        ],
    },
    "track5_pruning": {
        "attentions": ["linformer_attention", "adaptive_sparse_attention"],
        "compressors": [
            "slicegpt",
            "slicegpt_pca",
            "sparsegpt",
            "admm_pruner",
            "armor_pruner",
            "ast_pruner",
            "bonsai_pruner",
            "flap",
            "flat_llm",
            "lillama_compressor",
            "cachenotes_compressor",
            "come",
            "oats",
            "proxsparse",
            "soka",
            "liftr",
            "mamba_shedder",
            "sdrm",
        ],
    },
}


# =============================================================================
# BenchmarkResult Schema
# =============================================================================


@dataclass
class BenchmarkResult:
    """Standardized record containing metrics and diagnostics for an evaluation run."""

    track: str
    module_name: str
    module_type: str
    model_family: str
    dataset: str
    status: str
    error_message: str | None = None

    # Latency & Throughput
    forward_latency_ms: float = 0.0
    backward_latency_ms: float = 0.0
    total_step_latency_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    speedup_vs_baseline: float = 1.0

    # Memory & Parameters
    param_count: int = 0
    trainable_param_count: int = 0
    compression_ratio: float = 1.0
    peak_memory_mb: float = 0.0
    memory_savings_pct: float = 0.0

    # Accuracy & Loss
    loss: float = 0.0
    accuracy_pct: float = 0.0
    f1_score: float = 0.0
    perplexity: float = 0.0
    grad_norm: float = 0.0

    # Advanced metrics
    mcc: float = 0.0
    pearson: float = 0.0
    spearman: float = 0.0
    fidelity_score: float = 0.0
    flops_macs: float = 0.0
    throughput_tokens_sec: float = 0.0
    latency_variance_ms: float = 0.0
    ci_lower: float | None = None
    ci_upper: float | None = None
    p_value_vs_baseline: float | None = None
    mean_seed_acc: float | None = None
    std_seed_acc: float | None = None

    # Diagnostics
    calibration_used: bool = False
    calibration_source: str | None = None
    num_compressed_layers: int = 0
    num_fallback_layers: int = 0
    avg_reconstruction_error: float | None = None
    params_before: int = 0
    params_after: int = 0

    trajectories: dict[str, list[float]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary, sanitizing NaN/Inf floats for JSON serialization."""
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float) and not math.isfinite(v):
                d[k] = None
        return d


__all__ = [
    "BenchmarkResult",
    "TRACK_MODULES",
    "MODEL_TOKENIZER_MAPPING",
    "LM_DATASETS",
    "GLUE_DATASETS",
    "ACTIVATION_AWARE_MODULES",
    "COMPUTE_OPTIMIZERS",
    "KV_CACHE_OPTIMIZERS",
    "DEPRECATED_MODULES",
]
