Module neuralops.attentions.aegis_hybrid
========================================
Aegis-Hybrid Attention — Production-Hardened (v2).

Combines:
  1. Local sliding-window attention (chunked).
  2. Optional global linear attention via FAVOR+ style kernel features.
  3. Trainable LoRA adapters on top of frozen low-rank base.
  4. Learnable fusion gate between local and global branches.

Fixes (v2):
  AH-FIX-01  `FactorizedLinear.is_custom_attention = False`.
  AH-FIX-02  `use_global` is configurable; global path restored with proper
             learnable kernel feature map.
  AH-FIX-03  Fast `weight` property (no full matmul on each access).
  AH-FIX-04  GQA-aware SVD initialization with `repeat_interleave` pooling.
  AH-FIX-05  Robust GPT-2 Conv1D handling (checks layout at runtime).
  AH-FIX-06  Comprehensive mask normalization (2D/3D/4D).
  AH-FIX-07  Correct `past_len` derived from `past_key_value` shape, not
             from `N_kv - N` (which is wrong for cross-attention).
  AH-FIX-08  Full `inject_aegis_hybrid` with recursion + idempotency guard.
  AH-FIX-09  RoPE supported via `position_embeddings` or `self.rotary_emb`.
  AH-FIX-10  dtype-safe attention matmul.
  AH-FIX-11  `local_window` default consistent (256).
  AH-FIX-12  Storage accounting.
  AH-FIX-13  `output_hidden_states` / `return_dict` / `*args` supported.
  AH-FIX-14  `FactorizedLinear` uses `down`/`up` names for consistency
             with the rest of the framework, with backward-compat aliases
             `lin1`/`lin2`.
  AH-FIX-15  Optional activation whitening via `whitening` dict.

Functions
---------

`apply_rotary_pos_emb(q, k, cos, sin)`
:   

`inject_aegis_hybrid(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`AegisHybridAttention(config: Any | None = None, hidden_size: int | None = None, num_heads: int | None = None, pretrained_attn: nn.Module | None = None, rank: int | None = None, lora_rank: int | None = None, is_decoder: bool | None = None, whitening: dict[str, torch.Tensor] | None = None, **kwargs: Any)`
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

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
    :

    `out_lin`
    :

    `q_lin`
    :

    `query`
    :

    `v_lin`
    :

    `value`
    :

    ### Methods

    `storage_bytes(self) ‑> dict[str, int]`
    :

`AegisHybridConfig(hidden_size: int = 768, num_attention_heads: int = 12, head_dim: int = 64, num_key_value_heads: int | None = None, local_window: int = 256, rank: int = 128, lora_rank: int = 32, dropout: float = 0.1, use_adaptive_pruning: bool = False, gate_init_bias: float = 1.8, is_decoder: bool = False, use_global: bool = True, kernel_feature_dim: int = 128, kernel_seed: int = 1234, chunk_size: int = 64)`
:   AegisHybridConfig(hidden_size: 'int' = 768, num_attention_heads: 'int' = 12, head_dim: 'int' = 64, num_key_value_heads: 'int | None' = None, local_window: 'int' = 256, rank: 'int' = 128, lora_rank: 'int' = 32, dropout: 'float' = 0.1, use_adaptive_pruning: 'bool' = False, gate_init_bias: 'float' = 1.8, is_decoder: 'bool' = False, use_global: 'bool' = True, kernel_feature_dim: 'int' = 128, kernel_seed: 'int' = 1234, chunk_size: 'int' = 64)

    ### Instance variables

    `chunk_size: int`
    :   The type of the None singleton.

    `dropout: float`
    :   The type of the None singleton.

    `gate_init_bias: float`
    :   The type of the None singleton.

    `head_dim: int`
    :   The type of the None singleton.

    `hidden_size: int`
    :   The type of the None singleton.

    `is_decoder: bool`
    :   The type of the None singleton.

    `kernel_feature_dim: int`
    :   The type of the None singleton.

    `kernel_seed: int`
    :   The type of the None singleton.

    `local_window: int`
    :   The type of the None singleton.

    `lora_rank: int`
    :   The type of the None singleton.

    `num_attention_heads: int`
    :   The type of the None singleton.

    `num_key_value_heads: int | None`
    :   The type of the None singleton.

    `rank: int`
    :   The type of the None singleton.

    `use_adaptive_pruning: bool`
    :   The type of the None singleton.

    `use_global: bool`
    :   The type of the None singleton.

`FactorizedLinear(in_features: int, out_features: int, rank: int = 128, lora_rank: int = 32, bias: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Low-rank factorized linear + LoRA:
        y = (x @ V.T) @ U.T + LoRA(x) + b
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `bias: torch.Tensor | None`
    :

    `lin1`
    :

    `lin2`
    :

    `weight: torch.Tensor`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `reset_parameters(self) ‑> None`
    :

`KernelFeatureMap(head_dim: int, num_features: int = 128, seed: int = 1234)`
:   FAVOR+ style random Fourier features with deterministic seed.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.