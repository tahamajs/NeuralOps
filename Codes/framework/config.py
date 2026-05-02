"""Shared compression and method config used across attention modules and training."""
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CompressionConfig:
    # ── Attention type ──
    attn_type: str = "full"
    dropout: float = 0.1

    # ── Linformer / TokenCompressedAttention parameters ──
    k_early: int = 32
    k_mid: int = 16
    k_late: int = 12
    progressive: bool = True
    max_seq_len: int = 128

    # ── LEKA parameters ──
    num_landmarks: int = 16
    feature_dim: int = 128
    use_spherical_norm: bool = True
    kernel: str = "elu"
    use_gate: bool = True
    gate_bias: float = 0.0

    # ── SpectraV2 parameters ──
    block_size: int = 64
    top_k_inter: int = 16
    use_rel_pos_bias: bool = True
    rel_pos_max_distance: int = 128
    adaptive_span: bool = True
    max_span: float = 512.0
    rectify_interval: int = 4
    rff_features: int = 128
    sph_features: int = 128
    aks_enabled: bool = True

    # ── SVD specific ──
    svd_rank_qkv: int = 32
    svd_rank_out: int = 32
    ortho_reg_weight: float = 0.0
    use_orthogonal: bool = False

    # ── Method parameters ──
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    rdrop_alpha: float = 5.0
    r3f_epsilon: float = 1e-5
    llrd_decay: float = 0.95
    mixout_p: float = 0.9
    use_lion: bool = False
    dislora_beta: float = 0.5
    dislora_capacity_factor: float = 1.0

    # Fisher / GeLoRA
    fisher_num_batches: int = 20
    fisher_sample_size: int = 200
    gelora_margin: float = 1.0
    gelora_min_rank: int = 2
    gelora_max_rank: int = 32
    fisher_scale_max: float = 2.0

    # Training
    gradient_checkpointing: bool = False
    early_stopping_patience: Optional[int] = None
    max_grad_norm: float = 1.0