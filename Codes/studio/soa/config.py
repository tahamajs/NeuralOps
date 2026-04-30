# studio/soa/config.py
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
    k_rank: int = 64
    num_labels: int = 2

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        from transformers import AutoConfig
        hf = AutoConfig.from_pretrained(model_name)
        return cls(
            vocab_size=hf.vocab_size,
            hidden_size=hf.hidden_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            intermediate_size=hf.intermediate_size,
            **kwargs
        )