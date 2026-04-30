# framework/soa/config.py
"""
Configuration for SOA (SVD-Optimized Architecture) models.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SOAConfig:
    """
    Configuration class for SOA (SVD-Optimized Architecture) models.
    
    Attributes:
        vocab_size: Size of vocabulary
        max_position_embeddings: Maximum sequence length
        type_vocab_size: Size of token type vocabulary
        hidden_size: Hidden dimension size
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: Size of FFN intermediate layer
        hidden_act: Activation function name
        layer_norm_eps: LayerNorm epsilon
        dropout_prob: Dropout probability
        k_rank: Rank of the shared basis for low-rank attention
        num_labels: Number of output labels (for classification)
    """
    
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
    num_labels: int = 2       # For classification tasks
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """
        Create SOAConfig from a pretrained model's config.
        
        Args:
            model_name: HuggingFace model name
            **kwargs: Override values
            
        Returns:
            SOAConfig instance
        """
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_name)
        
        return cls(
            vocab_size=hf_config.vocab_size,
            max_position_embeddings=hf_config.max_position_embeddings,
            type_vocab_size=getattr(hf_config, 'type_vocab_size', 2),
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            intermediate_size=hf_config.intermediate_size,
            hidden_act=hf_config.hidden_act,
            layer_norm_eps=hf_config.layer_norm_eps,
            dropout_prob=hf_config.hidden_dropout_prob,
            **kwargs
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'max_position_embeddings': self.max_position_embeddings,
            'type_vocab_size': self.type_vocab_size,
            'hidden_size': self.hidden_size,
            'num_hidden_layers': self.num_hidden_layers,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'hidden_act': self.hidden_act,
            'layer_norm_eps': self.layer_norm_eps,
            'dropout_prob': self.dropout_prob,
            'k_rank': self.k_rank,
            'num_labels': self.num_labels,
        }