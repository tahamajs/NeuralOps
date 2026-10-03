Module neuralops.attentions.vspec
=================================
VSpec++ Attention: Variational Spectral Token-Adaptive Attention.
Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Applies rotary position embeddings safely handling even or odd dimensions.

`inject_vspec_attn(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any) ‑> neuralops.attentions.vspec.VSpecAttention | torch.nn.modules.module.Module | None`
:   Safe injection hook for VSpecAttention across diverse transformer architectures.

Classes
-------

`DynamicLowRankProjector(rank: int = 32, in_dim: int = 768, out_dim: int = 768, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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
    :   SVD-initialized low-rank projector with dynamic variational gate scaling and LoRA adapters.

    ### Methods

    `copy_lora_from(self, sub: Any) ‑> None`
    :   Preserve source LoRA adapters.

    `forward(self, x: torch.Tensor, gate: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `freeze_base_weights(self) ‑> None`
    :

    `set_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) ‑> None`
    :

`HeadGate(hidden_dim: int, n_heads: int)`
:   Learned per-example head importance scalars with masked pooling.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`TokenRouter(hidden_dim: int, keep_ratio: float = 0.75, min_tokens: int = 8)`
:   Top-k token selector based on learned saliency scores with guaranteed [CLS] retention.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None, is_decoder: bool = False) ‑> tuple[torch.Tensor, torch.Tensor]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`VSpecAttention(config: Any | None = None, rank: int = 64, pretrained_attn: nn.Module | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   VSpec++ Attention: Variational Spectral Token-Adaptive Attention Module.
    
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

    `k_proj`
    :

    `key`
    :

    `o_proj`
    :

    `out_proj`
    :

    `q_proj`
    :

    `query`
    :

    `v_proj`
    :

    `value`
    :

`VariationalSpectralGate(hidden_dim: int, rank: int)`
:   Input-dependent variational gate over truncated singular values with masked pooling.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.