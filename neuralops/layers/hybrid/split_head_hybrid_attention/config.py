from dataclasses import dataclass

@dataclass
class SplitHeadHybridConfig:
    """Configuration for SplitHeadHybridAttention."""

    svd_energy: float = 0.95
    min_rank: int = 4
    max_rank: int = 384
    num_local_heads: int = 8
    num_global_heads: int = 4
    local_window: int = 64
    softpick_temperature: float = 1.0
    rff_features: int = 128
    use_zero_sum_keys: bool = True
    use_adaptive_gate: bool = True
    gate_init_bias: float = 1.8
    dropout: float = 0.1
    is_decoder: bool = False
    rff_trainable: bool = True
    use_sdpa_fast_path: bool = True
    causal_chunk_size: int = 64
