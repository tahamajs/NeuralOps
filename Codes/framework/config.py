"""Shared compression config used across attention modules."""
from dataclasses import dataclass

@dataclass
class CompressionConfig:
    attn_type: str = "full"
    dropout: float = 0.1
    # Linformer / TokenCompressed Attention
    k_early: int = 32
    k_mid: int = 16
    k_late: int = 12
    progressive: bool = True
    max_seq_len: int = 128
    # LEKA
    num_landmarks: int = 16
    feature_dim: int = 128
    use_spherical_norm: bool = True
    kernel: str = "elu"
    use_gate: bool = True
    gate_bias: float = 0.0
    # SpectraV2
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
    # SVD specific
    svd_rank_qkv: int = 32
    svd_rank_out: int = 32
    ortho_reg_weight: float = 0.0
    use_orthogonal: bool = False
