import torch.nn as nn

class MockConfig:
    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get("hidden_size", kwargs.get("embed_dim", 64))
        self.embed_dim = self.hidden_size
        self.num_attention_heads = kwargs.get("num_attention_heads", kwargs.get("num_heads", 4))
        self.num_heads = self.num_attention_heads
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", 16)
        self.dropout = kwargs.get("dropout", kwargs.get("attention_dropout", 0.0))
        self.attention_dropout = self.dropout
        self.is_decoder = kwargs.get("is_decoder", False)
        self.model_type = kwargs.get("model_type", "bert")
        
        # Hyperparams
        self.rank = kwargs.get("rank", 16)
        self.lora_rank = kwargs.get("lora_rank", 16)
        self.lora_alpha = kwargs.get("lora_alpha", 32.0)
        self.page_size = kwargs.get("page_size", 4)
        self.k_experts = kwargs.get("k_experts", 2)
        self.svd_rank = kwargs.get("svd_rank", 4)
        self.alpha = kwargs.get("alpha", 1.5)
        
        self.max_position_embeddings = kwargs.get("max_seq_len", 128)
        self.max_seq_len = self.max_position_embeddings
        self.layer_norm_eps = kwargs.get("layer_norm_eps", 1e-12)
        self.window_size = kwargs.get("window_size", 4)
        self.block_size = kwargs.get("block_size", 4)
        self.top_k = kwargs.get("top_k", 2)

        for k, v in kwargs.items():
            setattr(self, k, v)

class DummyDistilBertAttention(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        
    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states, None)
