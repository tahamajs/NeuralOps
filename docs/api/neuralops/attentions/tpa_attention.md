Module neuralops.attentions.tpa_attention
=========================================
TPAttention: Tensor Product Attention with Factorised Q-K Projections.

Tensor-Product factorization of Q and K:
    W_q ≈ q_proj1 ⊗ q_proj2   (per head)
    W_q[h*hd + a*r + b, k] ≈ A[h, a, k] · B[h, b, k]
    W_k[h*hd + a*r + b, k] ≈ C[h, a, k] · D[h, b, k]

Functions
---------

`apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)`
:   

`inject_tpa(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: Any = None, **kwargs: Any) ‑> neuralops.attentions.tpa_attention.TPAttention | torch.nn.modules.module.Module | None`
:   

`repeat_kv(hidden_states: torch.Tensor, n_rep: int) ‑> torch.Tensor`
:   Expands key/value heads from num_key_value_heads to num_attention_heads.
    Equivalent to torch.repeat_interleave(hidden_states, dim=1, repeats=n_rep)
    Shape: (batch, num_key_value_heads, seq_len, head_dim) -> (batch, num_attention_heads, seq_len, head_dim)
    For standard MHA where n_rep <= 1, returns hidden_states directly (zero-copy).

`tp_inject(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: Any = None, **kwargs: Any) ‑> neuralops.attentions.tpa_attention.TPAttention | torch.nn.modules.module.Module | None`
:   

Classes
-------

`TPAttention(config: Any | None = None, rank: int = 8, pretrained_attn: nn.Module | None = None, lora_rank: int = 0, lora_alpha: float = 1.0, is_decoder: bool | None = None, max_cp_error: float = 0.35, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, *args, **kwargs) ‑> Callable[..., typing.Any]`
    :   Must return (output, attention_weights_or_None).

`TPA(config: Any | None = None, rank: int = 8, pretrained_attn: nn.Module | None = None, lora_rank: int = 0, lora_alpha: float = 1.0, is_decoder: bool | None = None, max_cp_error: float = 0.35, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.