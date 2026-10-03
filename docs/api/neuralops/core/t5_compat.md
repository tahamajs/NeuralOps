Module neuralops.core.t5_compat
===============================
T5 Architecture Compatibility Module.
Provides robust support for T5 relative position bias, encoder-decoder cross-attention,
and numerically stable FP16 forward execution for custom attention modules.

Functions
---------

`compute_t5_bias(query_length: int, key_length: int, relative_attention_bias: torch.nn.modules.sparse.Embedding | None, is_decoder: bool = False, num_buckets: int = 32, max_distance: int = 128, device: torch.device | None = None) ‑> torch.Tensor | None`
:   Compute binned relative position bias table [1, num_heads, query_length, key_length].

`get_t5_dims(config: Any) ‑> tuple[int, int, int]`
:   Extract (hidden_size, num_heads, head_dim) cleanly from T5Config or standard config.

`is_t5_model(config: Any) ‑> bool`
:   Detect if a configuration belongs to the T5 family.

`relative_position_bucket(relative_position: torch.Tensor, bidirectional: bool = True, num_buckets: int = 32, max_distance: int = 128) ‑> torch.Tensor`
:   Translate relative positions into bucket IDs matching T5Attention._relative_position_bucket.

Classes
-------

`T5AttentionMixin()`
:   Mixin providing full T5 encoder-decoder cross-attention and relative position bias
    handling for any attention class.

    ### Methods

    `compute_bias(self, query_length: int, key_length: int, device: torch.device | None = None) ‑> torch.Tensor | None`
    :

    `copy_t5_metadata(self, old_attn: Any) ‑> None`
    :   Copy relative position bias embeddings and metadata from native T5 attention.

    `forward_t5_base(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, key_value_states: torch.Tensor | None = None, position_bias: torch.Tensor | None = None, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor]`
    :   Numerically stable T5 forward execution supporting:
        - Encoder self-attention
        - Decoder self-attention
        - Decoder cross-attention
        - Relative position bias computation and propagation

    `init_t5_support(self, config: Any) ‑> None`
    :   Initialize T5 attributes on module.