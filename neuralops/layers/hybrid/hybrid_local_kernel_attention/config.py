from dataclasses import dataclass

@dataclass
class HybridLocalKernelConfig:
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_heads: int = 12
    head_dim: int = 64
    num_key_value_heads: int = 12
    local_window: int = 64
    rank: int = 384
    kernel_feature_dim: int = 128
    dropout: float = 0.1
    use_adaptive_pruning: bool = True
    gate_init_bias: float = 1.8
    is_decoder: bool = False
