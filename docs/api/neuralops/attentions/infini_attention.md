Module neuralops.attentions.infini_attention
============================================
Infini-Attention with Compressive Memory and Sliding Window — Fixed (v3).

Reference: Munkhdalai et al., "Leave No Context Behind: Efficient Infinite
Context Transformers with Infini-attention", 2024.

Core mechanism:
    A_local = sliding-window softmax attention (with causal mask for decoder)
    M_t     = M_{t-1} + g_t · σ(K_t)^T V_t         (compressive memory update)
    z_t     = z_{t-1} + g_t · σ(K_t)                 (normalizer)
    A_mem   = σ(Q_t) M_t / (σ(Q_t) z_t + ε)          (memory read)
    output  = β · A_mem + (1-β) · A_local             (fusion)

where σ(x) = ELU(x) + 1.

Fixes (v3):
  INF3-FIX-01  Forward signature now matches BERT/RoBERTa/DistilBERT
               positional arg order exactly, so HF calls bind correctly.
               `output_attentions` is the 7th positional arg (BERT style);
               DistilBERT passes it as keyword. Both work.
  INF3-FIX-02  `layer_idx` accepted as keyword with default 0.
  INF3-FIX-03  All internal parameters (`beta`, `w_mem.weight`, `w_mem.bias`)
               are explicitly initialized on the target device/dtype
               BEFORE the first forward pass, so no device mismatch.
  INF3-FIX-04  `_compressive_memory_read` precomputes `g · φ(k)` outside the
               loop and uses a batched outer product; avoids `.to()` churn.
  INF3-FIX-05  Memory state `M`, `Z` explicitly declared fp32 to avoid
               autocast-related dtype drift.
  INF3-FIX-06  Idempotency guard checks both source and parent layer.
  INF3-FIX-07  Recursion guard in `inject_infini` (depth limit).
  INF3-FIX-08  Sliding-window + causal + padding masks combined in one
               `masked_fill` chain.
  INF3-FIX-09  `nan_to_num` on all outputs (scores, attn, context, memory).
  INF3-FIX-10  GQA `repeat_kv` applied before memory read, so `v` matches
               the head layout used in the memory loop.
  INF3-FIX-11  Storage accounting uses a single sum over real parameters.
  INF3-FIX-12  `copy_weights` raises a descriptive error on irreconcilable
               shape mismatch instead of silently truncating.
  INF3-FIX-13  Proper handling of `head_mask` for 1D/2D/3D shapes.
  INF3-FIX-14  `comp_config` extraction tolerant of dict/None/namespace.

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_infini(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any) ‑> torch.nn.modules.module.Module | None`
:   

Classes
-------

`InfiniAttention(config: Any | None = None, rank: int | None = None, local_window: int = 128, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Infini-Attention: sliding-window local attention + compressive memory.
    
    For decoders, the output is a per-head convex combination of:
      - A_local: sliding-window softmax attention
      - A_mem:   read from a transient (per-forward) compressive memory state
    For encoders, it degenerates to full softmax attention (memory is skipped).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
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

`LoRALinear(in_features, out_features, bias=True, lora_rank=16, lora_alpha=32.0, freeze_base=False, device=None, dtype=None)`
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

    ### Methods

    `forward(self, x) ‑> Callable[..., typing.Any]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `freeze_base_weights(self)`
    :

    `reset_parameters(self)`
    :