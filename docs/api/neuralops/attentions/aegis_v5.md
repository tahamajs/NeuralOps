Module neuralops.attentions.aegis_v5
====================================
AEGIS-Attention V5 (Corrected): Adaptive Enhanced Global-Local Integrated Sparse Attention.

Architecture:
  * FactorisedLinear: SVD-factorized linear layers with energy-based rank selection + LoRA.
  * Split-head attention: some heads perform local sliding-window attention,
    other heads perform global kernel linear attention (RFF / FAVOR+).
  * Adaptive gate: blends local-head outputs with the mean of global-head outputs.
  * SDPA fast path: used for very short sequences (N_kv <= local_window) where the
    split-head mechanism provides no benefit.

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Apply RoPE. Caller must slice cos/sin to the current window (past_len : past_len+N).

`causal_kernel_linear_attention(Q_feat: torch.Tensor, K_feat: torch.Tensor, V: torch.Tensor, chunk_size: int = 64) ‑> torch.Tensor`
:   V5-FIX-03: causal kernel linear attention using chunked cumsum.
    
    Inputs:
        Q_feat: (B, H, N, F) — query features (after exp)
        K_feat: (B, H, N, F) — key features (after exp)
        V:      (B, H, N, D) — values
    Output:
        context: (B, H, N, D)
    
    Memory: O(B*H*chunk_size*F*D) at peak, vs O(B*H*N*F*D) for the naive version.
    Time:   O(N * F * D) forward.

`inject_aegis_into_distilbert(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

`inject_aegis_v5(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

`kernel_linear_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, rff_proj: torch.Tensor, use_zero_sum: bool = True, max_clamp: float = 11.0) ‑> torch.Tensor`
:   Non-causal global kernel linear attention (FAVOR+ style).
    
    rff_proj: (H_g, D, F) random Fourier features for global heads.
    Returns context of shape (B, H_g, N, D).

`local_window_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, window: int, softpick_temperature: float = 1.0, token_valid: torch.Tensor | None = None, is_decoder: bool = False, past_len: int = 0) ‑> torch.Tensor`
:   Sliding-window attention.
    
    V5-FIX-02: `past_len` shifts the query positions so that the window is
    measured in absolute token positions. In prefill, past_len=0. In decode
    with a KV-cache of length L and N new tokens, past_len=L.
    
    V5-FIX-09: `window` is the full width of the sliding window. The effective
    radius used for masking is `window // 2`, so a query at position q attends
    to keys in [q - window//2, q + window//2].

`temperature_softmax(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) ‑> torch.Tensor`
:   Softmax with temperature. Stable under fp32 computation (V5-FIX-08: renamed).

Classes
-------

`AegisAttention(model_config: Any = None, config: Any = None, aegis_cfg: neuralops.attentions.aegis_v5.AegisConfigV5 | None = None, pretrained_attn: torch.nn.modules.module.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
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

    ### Instance variables

    `c_proj: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `dense: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `k_lin: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `key: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `o_proj: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `out_lin: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `q_lin: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `query: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `v_lin: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `value: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`AegisAttentionV5(model_config: Any = None, config: Any = None, aegis_cfg: neuralops.attentions.aegis_v5.AegisConfigV5 | None = None, pretrained_attn: torch.nn.modules.module.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
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

    ### Instance variables

    `c_proj: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `dense: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `k_lin: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `key: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `o_proj: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `out_lin: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `q_lin: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `query: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `v_lin: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `value: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    ### Methods

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`AegisConfigV5(svd_energy: float = 0.95, min_rank: int = 4, max_rank: int = 384, num_local_heads: int = 8, num_global_heads: int = 4, local_window: int = 64, softpick_temperature: float = 1.0, rff_features: int = 128, use_zero_sum_keys: bool = True, use_adaptive_gate: bool = True, gate_init_bias: float = 1.8, dropout: float = 0.1, is_decoder: bool = False, rff_trainable: bool = False, use_sdpa_fast_path: bool = True, causal_chunk_size: int = 64)`
:   Configuration for AegisAttention V5.

    ### Instance variables

    `causal_chunk_size: int`
    :   The type of the None singleton.

    `dropout: float`
    :   The type of the None singleton.

    `gate_init_bias: float`
    :   The type of the None singleton.

    `is_decoder: bool`
    :   The type of the None singleton.

    `local_window: int`
    :   The type of the None singleton.

    `max_rank: int`
    :   The type of the None singleton.

    `min_rank: int`
    :   The type of the None singleton.

    `num_global_heads: int`
    :   The type of the None singleton.

    `num_local_heads: int`
    :   The type of the None singleton.

    `rff_features: int`
    :   The type of the None singleton.

    `rff_trainable: bool`
    :   The type of the None singleton.

    `softpick_temperature: float`
    :   The type of the None singleton.

    `svd_energy: float`
    :   The type of the None singleton.

    `use_adaptive_gate: bool`
    :   The type of the None singleton.

    `use_sdpa_fast_path: bool`
    :   The type of the None singleton.

    `use_zero_sum_keys: bool`
    :   The type of the None singleton.

`FactorisedLinear(in_features_or_weight: Any, out_features_or_bias: Any | None = None, energy: float = 0.95, min_rank: int = 4, max_rank: int = 384, rank: int | None = None, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, **kwargs: Any)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing them to be nested in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
    
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will also have their
    parameters converted when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   SVD-factorized linear layer with energy-based rank selection and LoRA adapters.
        
        Forward: y = up(down(x)) + (B @ A)(x) * (lora_alpha / lora_rank)
        Base SVD weights (down, up) are frozen; only LoRA (A, B) is trainable.
        
        Note on naming: `lora_A` has shape (rank, in) and `lora_B` has shape (out, rank),
        matching the PEFT convention. This differs from AegisAttentionV6 which uses the
        opposite convention.

    ### Instance variables

    `bias: torch.Tensor | None`
    :

    `weight: torch.Tensor`
    :

    ### Methods

    `decompose_from_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None, energy: float | None = None) ‑> None`
    :

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `freeze_base_weights(self) ‑> None`
    :

    `reset_parameters(self) ‑> None`
    :