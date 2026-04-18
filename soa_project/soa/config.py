class SOAConfig:
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        k_rank: int = 64,        # Rank for low-rank approximation
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.k_rank = k_rank