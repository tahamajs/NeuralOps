Module neuralops.attentions.moe_deltakv_paged
=============================================
MoE-DeltaKV-Paged Hybrid Attention — Production-Hardened v2.

Combines:
  1. Mixture-of-Block-Attention (MoBA) with learnable page routing.
  2. Paged Delta-KV temporal compression with adaptive quantization.
  3. Optional low-rank factorization of temporal deltas.

Fixes (v2):
  MDP-FIX-01  **Adaptive page_size**: if `page_size >= seq_len`, paging is
              disabled (falls back to full attention for that layer).
  MDP-FIX-02  **STE quantization now has a reconstruction-error guard.**
              If delta-KV reconstruction error exceeds `max_delta_error`,
              we skip compression and use the raw K/V.
  MDP-FIX-03  **`aux_loss` is now an `nn.Parameter` of shape `(1,)`** so it
              travels with the model and can be added to the training loss.
  MDP-FIX-04  **MoE routing now runs on encoder models too** (removed the
              hard `is_causal` short-circuit).
  MDP-FIX-05  **`is_nested_encoder` for DistilBERT correctly set to `True`**,
              and `out_proj` skipped during forward for nested encoders.
  MDP-FIX-06  **Idempotency guard** in `inject_moe_deltakv_paged`.
  MDP-FIX-07  **Storage accounting** in `storage_bytes()`.
  MDP-FIX-08  **`LoRALinear.forward` is dtype-safe.**
  MDP-FIX-09  **Per-projection try/except in `copy_weights`.**
  MDP-FIX-10  **Cleaner page routing** with causal masking that matches the
              `is_causal` flag correctly.
  MDP-FIX-11  **CLS/page-0 bias is only applied when causal masking allows it.**
  MDP-FIX-12  **`q_bits` and `svd_rank` are validated at construction.**
  MDP-FIX-13  **Forward returns `(out, attn_weights)`; second item is always
              either a tensor or `None` — never a tuple.**
  MDP-FIX-14  **`hidden_size`, `num_heads`, `head_dim` inferred consistently
              across all architectures.**

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_moe_deltakv_paged(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any)`
:   

`low_rank_delta_compress(delta: torch.Tensor, rank: int, bits: int) ‑> tuple[torch.Tensor, float]`
:   Low-rank + STE-quantize a temporal delta tensor.
    
    Returns (compressed, relative_reconstruction_error).

`simulate_paged_delta_kv(K: torch.Tensor, V: torch.Tensor, rank: int, bits: int, max_delta_error: float = 0.35) ‑> tuple[torch.Tensor, torch.Tensor, float, float]`
:   Delta-KV temporal compression with reconstruction-error guard.
    
    MDP-FIX-02: if the reconstruction error exceeds `max_delta_error`, the
    raw K/V is returned instead of the compressed version.
    
    Returns (K_hat, V_hat, err_K, err_V).

`ste_quantize(x: torch.Tensor, bits: int = 4) ‑> torch.Tensor`
:   Symmetric uniform quantization with straight-through estimator.
    
    MDP-FIX-02: returns the quantized tensor; caller is responsible for
    checking reconstruction error and deciding whether to keep it.

Classes
-------

`LoRALinear(in_features: int, out_features: int, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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

    `freeze_base_weights(self) ‑> None`
    :

    `reset_parameters(self) ‑> None`
    :

`MoEDeltaKVPagedAttention(config: Any | None = None, page_size: int = 16, k_experts: int = 4, svd_rank: int = 32, q_bits: int = 4, max_delta_error: float = 0.35, lora_rank: int = 16, lora_alpha: float = 32.0, enable_moe: bool = True, enable_deltakv: bool = True, is_decoder: bool | None = None, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   MoE-DeltaKV-Paged Hybrid Attention.
    
    Two execution paths:
      * Full dense attention when `num_pages <= 1` (page_size >= seq_len)
        OR when `enable_moe=False`.
      * Paged MoE attention otherwise.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `k_proj`
    :

    `o_proj`
    :

    `out_lin`
    :

    `q_lin`
    :

    `q_proj`
    :

    `v_lin`
    :

    `v_proj`
    :

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `storage_bytes(self) ‑> dict[str, int]`
    :