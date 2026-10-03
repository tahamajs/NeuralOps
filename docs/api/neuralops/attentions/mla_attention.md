Module neuralops.attentions.mla_attention
=========================================
Multi-Head Latent Attention (MLA) – DeepSeek-V2 (Production-Hardened v2).

Paper: https://arxiv.org/abs/2405.04434

Reference implementation for DeepSeek-V2, adapted for multi-architecture
deployment across DistilBERT, RoBERTa, GPT-2, Qwen/LLaMA.

Note on encoder-only models:
  MLA was designed for decoder inference with KV-cache compression. On
  encoder-only models (BERT/RoBERTa/DistilBERT) it acts as a low-rank
  bottleneck on K/V. If the reconstruction error is too high the model
  loses accuracy. This version validates reconstruction error and falls
  back to the original attention when needed.

Fixes (v2):
  MLA-FIX-01  **Removed erroneous `norm_scale` on K/V.** The SVD low-rank
              approximation already has norm `||S[:r]||_F`; multiplying by
              `||W||_F / ||S[:r]||_F` scales the reconstruction back UP to
              the original norm, making attention scores explode.
  MLA-FIX-02  **Added `q_lin` / `k_lin` / `v_lin` / `out_lin` aliases** so
              DistilBERT's `DistilBertAttention` container can be replaced
              without breaking the forward pass.
  MLA-FIX-03  **Added `fallback_to_original` + `max_error`** so the module
              refuses to install itself if K/V reconstruction error is too
              high.
  MLA-FIX-04  **Added explicit K/V reconstruction-error validation** with
              logging before installing.
  MLA-FIX-05  **Robust `super().__init__`** with None-safe config.
  MLA-FIX-06  **Idempotency guard** in `copy_weights` (early exit if
              already initialized from the same source).
  MLA-FIX-07  **Fixed `is_nested_encoder` determination** using the source
              attention class (not just attribute probing).
  MLA-FIX-08  **Storage accounting** via `storage_bytes()`.
  MLA-FIX-09  **`is_custom_attention = False` on helper layers**
              (`MLALoRALinear` is a linear layer, not attention).
  MLA-FIX-10  **`dense`/`c_proj`/`o_proj` aliases** for GPT-2 / BART
              compatibility.
  MLA-FIX-11  **LoRA forward is dtype-safe**.
  MLA-FIX-12  **Injector returns the new module** so counters work.

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_mla(model: nn.Module | None = None, layer: nn.Module | None = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 1, old_attn: Any = None, **kwargs: Any) ‑> neuralops.attentions.mla_attention.MLAAttention | torch.nn.modules.module.Module | None`
:   

`inject_mla_attention(model: nn.Module | None = None, layer: nn.Module | None = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 1, old_attn: Any = None, **kwargs: Any) ‑> neuralops.attentions.mla_attention.MLAAttention | torch.nn.modules.module.Module | None`
:   

Classes
-------

`LoRALinear(in_features: int, out_features: int, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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

`MLAAttention(config: Any | None = None, rank: int | None = 128, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, max_error: float = 0.35, fallback_to_original: bool = True, **kwargs: Any)`
:   Multi-Head Latent Attention (MLA) — DeepSeek-V2 style.
    
    Idea: compress K and V into a low-rank latent `c_kv` of size `rank`,
    then up-project back to per-head K and V.
    
        c_kv = kv_down(x)             # (B, N, rank)
        k    = k_up(c_kv)              # (B, N, num_kv_heads * head_dim)
        v    = v_up(c_kv)              # (B, N, num_kv_heads * head_dim)
    
    Query is projected with a LoRA-augmented linear. K/V down/up projections
    are initialized via joint SVD of [W_k; W_v] and frozen.
    
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

    `triton_kernel: str`
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

    `is_compatible(self) ‑> bool`
    :   Return True if reconstruction error is within tolerance.

    `storage_bytes(self) ‑> dict[str, int]`
    :