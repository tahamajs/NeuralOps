Module neuralops.attentions.gqa_attention
=========================================
Grouped-Query Attention (GQA) — v3, rank-aware.

Interpolates between Multi-Head Attention (MHA) and Multi-Query Attention (MQA)
by dividing H_Q query heads into G groups, where each group shares a single
Key-Value head pair (H_KV = G ≤ H_Q).

Fixes (v3):
  GQA-FIX-01  Causal mask via past_len, always applied for decoders.
  GQA-FIX-02  Removed output padding/truncation hack.
  GQA-FIX-03  Deterministic layer detection.
  GQA-FIX-04  Removed thread-unsafe global injection depth.
  GQA-FIX-05  Idempotency guard in injector.
  GQA-FIX-06  Robust mask normalization (2D/3D/4D).
  GQA-FIX-07  Consistent is_nested_encoder propagation.
  GQA-FIX-08  Storage accounting.
  GQA-FIX-09  Partial-RoPE support.
  GQA-FIX-10  Conv1D bias handled correctly.
  GQA-FIX-11  GQA-aware copy_weights with repeat_interleave pooling.
  GQA-FIX-12  supports_rope flag.
  GQA-FIX-13  dtype-safe RoPE.
  GQA-FIX-14  Removed redundant double softmax.
  GQA-FIX-15  get_ortho_loss for framework compatibility.
  GQA-FIX-16  **`rank` now maps to `num_kv_heads` when passed via injector.**
              Rank sweep actually produces different compression levels.
  GQA-FIX-17  **`_unpack_weight_bias` handled for LoRA-wrapped modules.**
  GQA-FIX-18  **Numerically safe padding mask path.**

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_gqa(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, rank: int | None = None, num_kv_heads: int | None = None, **kwargs: Any) ‑> Any | None`
:   

Classes
-------

`GroupedQueryAttention(config: Any | None = None, rank: int | None = None, dim: int | None = None, num_heads: int | None = None, num_kv_heads: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
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

    `get_ortho_loss(self)`
    :

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