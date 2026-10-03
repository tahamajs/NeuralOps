"""
Complete configuration dataclasses for all framework components.

Usage:
    cfg = Config()
    cfg.model.name = "distilbert-base-uncased"
    cfg.compression.rank = 64
"""

from dataclasses import dataclass, field
from typing import Any

from neuralops.utils.paths import get_glue_dir

# ---------------------------------------------------------------------------
#  Sub-configs for specific components
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1


@dataclass
class RDropConfig:
    alpha: float = 5.0


@dataclass
class R3FConfig:
    epsilon: float = 1e-5


@dataclass
class FisherConfig:
    num_batches: int = 20
    sample_size: int = 200
    scale_max: float = 2.0


@dataclass
class GeLoRAConfig:
    margin: float = 1.0
    min_rank: int = 2
    max_rank: int = 32


@dataclass
class LLRDConfig:
    decay: float = 0.95


@dataclass
class MixoutConfig:
    p: float = 0.9


@dataclass
class DisLoRAConfig:
    beta: float = 0.5
    capacity_factor: float = 1.0


@dataclass
class MethodConfig:
    """Training method selection and its sub‑configs."""

    method: str = (
        "baseline"  # baseline, lora, dora, dislora, rdrop, r3f, fisher, gelora, llrd, mixout
    )
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    rdrop: RDropConfig = field(default_factory=RDropConfig)
    r3f: R3FConfig = field(default_factory=R3FConfig)
    fisher: FisherConfig = field(default_factory=FisherConfig)
    gelora: GeLoRAConfig = field(default_factory=GeLoRAConfig)
    llrd: LLRDConfig = field(default_factory=LLRDConfig)
    mixout: MixoutConfig = field(default_factory=MixoutConfig)
    dislora: DisLoRAConfig = field(default_factory=DisLoRAConfig)


@dataclass
class HardwareConfig:
    device: str = "auto"  # auto, cuda, mps, cpu
    force_cpu_for_svd: bool = True
    enable_fp16: bool = False
    enable_bf16: bool = False
    num_workers: int = 4
    pin_memory: bool = False
    torch_compile: bool = False
    empty_cache_every_step: bool = False


@dataclass
class DatasetConfig:
    """Description of a single dataset for an experiment sweep."""

    name: str
    path: str
    task_type: str = "single"  # single, pair, regression, mc
    num_choices: int = 0  # for multi‑choice
    num_labels: int = 2


@dataclass
class ModelConfig:
    """Model loading configuration."""

    name: str = "distilbert-base-uncased"
    num_labels: int = 2
    attention_type: str = "full"  # registry key
    use_fast_tokenizer: bool = True
    trust_remote_code: bool = False
    cache_dir: str = "./cache"
    fix_layernorm_naming: bool = True
    variants: list[str] = field(
        default_factory=list
    )  # e.g., ["standard", "compressed"]


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""

    local_glue_path: str = str(get_glue_dir())
    max_seq_length: int = 128
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    num_proc: int = 4
    streaming: bool = False
    remove_columns: list[str] = field(default_factory=lambda: ["label"])
    split_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "train": "train",
            "validation": "validation",
            "validation_matched": "validation_matched",
            "validation_mismatched": "validation_mismatched",
        }
    )


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    optimizer: str = "adamw"
    early_stopping_patience: int | None = None
    early_stopping_threshold: float = 0.001
    save_strategy: str = "epoch"
    save_steps: int = 500
    save_total_limit: int = 2
    save_only_model: bool = True
    eval_strategy: str = "epoch"
    eval_steps: int = 500
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16: bool = False
    bf16: bool = False
    report_to: str = "none"


@dataclass
class CompressionConfig:
    """
    All parameters related to attention injection, low‑rank compression,
    sparsity, hybrid heads, etc.
    These fields are read by the injection functions via getattr(comp_config, …).
    """

    # ── General ────────────────────────────────────────
    attn_type: str = "full"
    dropout: float = 0.1
    rank: int = 32  # generic rank (used by many modules)
    svd_rank: int = 32  # SVD rank
    svd_energy: float = 0.95  # energy retention for SVD truncation
    min_rank: int = 4
    max_rank: int = 64

    # ── LoRA / method‑specific ─────────────────────────
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    rdrop_alpha: float = 5.0
    r3f_epsilon: float = 1e-5
    llrd_decay: float = 0.95
    mixout_p: float = 0.9
    dislora_beta: float = 0.5
    dislora_capacity_factor: float = 1.0
    fisher_num_batches: int = 20
    fisher_sample_size: int = 200
    fisher_scale_max: float = 2.0
    gelora_margin: float = 1.0
    gelora_min_rank: int = 2
    gelora_max_rank: int = 32

    # ── Attention common fields ────────────────────────
    block_size: int = 64
    local_window: int = 64
    top_k_inter: int = 4
    num_landmarks: int = 16
    feature_dim: int = 128
    rff_features: int = 128
    k_early: int = 32
    k_mid: int = 16
    k_late: int = 12
    use_gate: bool = True
    progressive: bool = True
    max_seq_len: int = 512
    low_rank_cache_size: int = 16  # LoLA
    cur_rank: int = 16  # PHANTOM
    lsh_rounds: int = 4
    lsh_bucket_size: int = 32
    window_size: int = 128
    compression_rank: int = 64
    use_adaptive_gate: bool = True
    gate_bias: float = 2.0
    softpick_temperature: float = 1.0
    rectify_interval: int = 4
    aks_enabled: bool = True
    rel_pos_max_distance: int = 128
    use_rel_pos_bias: bool = True
    adaptive_span: bool = True
    max_span: float = 512.0
    num_kernel_heads: int = 8
    num_full_heads: int = 4
    soa_rank: int = 32
    svd_rank_qkv: int = 32
    svd_rank_out: int = 32
    num_local_heads: int = 8
    num_global_heads: int = 4
    num_experts: int = 8
    top_k: int = 2
    ratio: int = 9  # hybrid attention / Mamba ratio
    n_heads: int | None = None
    n_layers: int | None = None
    use_full_attention: bool = False
    use_fp8: bool = False

    # Additional fields used by some compressors
    use_lion: bool = False
    gradient_checkpointing: bool = False
    early_stopping_patience: int | None = None
    max_grad_norm: float = 1.0


@dataclass
class SweepConfig:
    """Hyperparameter sweep configuration."""

    name: str = "sweep"
    algorithm: str = "tpe"  # tpe, random, grid, cmaes
    direction: str = "maximize"
    metric: str = "eval_accuracy"
    max_trials: int = 20
    parallel_workers: int = 1
    pruner: str = "hyperband"
    search_space: dict[str, Any] = field(default_factory=dict)
    base_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Multi‑run experiment grid (SVD ranks, datasets, models)."""

    base_models: dict[str, str] = field(
        default_factory=lambda: {"bert": "bert-base-uncased"}
    )
    svd_ranks: list[int] = field(default_factory=lambda: [8, 16, 32])
    datasets: list[DatasetConfig] = field(default_factory=list)
    min_samples: int = 10
    max_samples: int = 200
    batch_size: int = 4
    epochs: int = 1
    learning_rate: float = 2e-5
    max_length: int = 128
    val_split: float = 0.2
    seed: int = 42
    gradient_accumulation_steps: int = 1
    save_checkpoint_interval: int = 10
    output_dir: str = "./results"
    plots_dir: str = "./plots/comprehensive"
    log_dir: str = "./logs"
    selected_models: list[str] = field(default_factory=list)
    selected_datasets: list[str] = field(default_factory=list)
    selected_ranks: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
#  Top‑level configuration aggregator
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Master configuration that holds all sub‑configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
