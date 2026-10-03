from dataclasses import dataclass


@dataclass
class CompressionConfig:
    """Unified configuration for attention injection and compression."""

    attn_type: str = "full"
    dropout: float = 0.1

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    # R‑Drop / R3F
    rdrop_alpha: float = 5.0
    r3f_epsilon: float = 1e-5

    # LLRD
    llrd_decay: float = 0.95

    # Mixout
    mixout_p: float = 0.9

    # DisLoRA
    dislora_beta: float = 0.5
    dislora_capacity_factor: float = 1.0

    # Fisher
    fisher_num_batches: int = 20
    fisher_sample_size: int = 200
    fisher_scale_max: float = 2.0

    # GeLoRA
    gelora_margin: float = 1.0
    gelora_min_rank: int = 2
    gelora_max_rank: int = 32

    # Optimizer
    use_lion: bool = False

    # Training
    gradient_checkpointing: bool = False
    early_stopping_patience: int | None = None
    max_grad_norm: float = 1.0

    # Attention‑specific common fields
    rank: int = 32  # general rank for SVD/LoRA
    svd_rank: int = 32  # SVD rank for compressed attention
    svd_energy: float = 0.95  # energy threshold for SVD
    block_size: int = 64  # block size for block-sparse attention
    local_window: int = 64  # window size for sliding window attention
    top_k_inter: int = 4  # top-k inter-block tokens
    num_landmarks: int = 16
    feature_dim: int = 128
    rff_features: int = 128  # random Fourier features dimension
    k_early: int = 32
    k_mid: int = 16
    k_late: int = 12
    use_gate: bool = True
    progressive: bool = True
    max_seq_len: int = 512  # was incomplete; set default 512
    low_rank_cache_size: int = 16  # for LoLA
    cur_rank: int = 16  # for PHANTOM
    lsh_rounds: int = 4
    lsh_bucket_size: int = 32
    window_size: int = 128  # for sliding window
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
    min_rank: int = 4
    max_rank: int = 64
    num_local_heads: int = 8
    num_global_heads: int = 4
    num_experts: int = 8
    top_k: int = 2
    ratio: int = 9  # for hybrid models
    n_heads: int | None = None  # can override
    n_layers: int | None = None
    use_full_attention: bool = False
    use_fp8: bool = False
