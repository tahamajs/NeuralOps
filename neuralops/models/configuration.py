"""Configuration class for NeuralOps Transformers-compatible models."""

from __future__ import annotations

from typing import Any
from transformers.configuration_utils import PretrainedConfig

from neuralops.models.hybrid_spec import normalize_hybrid_spec


__all__ = ["NeuralOpsConfig"]


class NeuralOpsConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `NeuralOpsModel`.

    Inherits from `PretrainedConfig` to enable seamless integration with
    HuggingFace AutoConfig, serialization, and model Hub features.
    """

    model_type = "neuralops"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        intermediate_size: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, Any] | None = None,
        attention_type: str = "svd_linear_attention",
        compression_type: str | None = None,
        attn: Any = None,
        compression: Any = None,
        svd_rank: int = 64,
        norm_type: str = "rmsnorm",
        mlp_type: str = "swiglu",
        dropout: float = 0.0,
        fuse_linear_cross_entropy: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize NeuralOpsConfig with full architectural parameters."""
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        )
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else int(hidden_size * 8 / 3)
        )
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_type = attention_type
        self.compression_type = compression_type
        self.attn = attn
        self.compression = compression
        self.svd_rank = svd_rank
        self.norm_type = norm_type
        self.mlp_type = mlp_type
        self.dropout = dropout
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self._validate()

    def _validate(self) -> None:
        """Validate architectural parameters."""
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {self.num_hidden_layers}")
        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {self.num_attention_heads}")
        if self.hidden_size % self.num_attention_heads != 0 and self.head_dim is None:
            raise ValueError("hidden_size must be divisible by num_attention_heads when head_dim is None")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads for GQA")

    def get_layer_attn_spec(self, layer_idx: int) -> dict[str, Any]:
        """Return the attention specification dictionary for a given layer index."""
        specs = normalize_hybrid_spec(
            self.attn,
            num_layers=self.num_hidden_layers,
            default_type=self.attention_type,
        )
        return specs[layer_idx]

    def get_layer_compression_spec(self, layer_idx: int) -> dict[str, Any] | None:
        """Return the compression specification dictionary for a given layer index."""
        if self.compression is None and self.compression_type is None:
            return None
        default_comp = self.compression_type or "none"
        specs = normalize_hybrid_spec(
            self.compression,
            num_layers=self.num_hidden_layers,
            default_type=default_comp,
        )
        layer_spec = specs[layer_idx]
        if layer_spec.get("type") in ("none", None):
            return None
        return layer_spec
