from dataclasses import dataclass

@dataclass
class SOAConfig:
    vocab_size: int = 30522
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-12
    dropout_prob: float = 0.1
    k_rank: int = 64          # Rank of the shared basis