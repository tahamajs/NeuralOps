Module neuralops.attentions.nexus
=================================
NEXUS: Neural Efficient X-scoring Unified Sparse Attention.

=============================================================
Hybrid sparse attention architecture combining multi-scale sparsity mechanisms:
 1. XAttention antidiagonal block scoring (ICML 2025) - training-free block routing
 2. SeerAttention learnable gating (NeurIPS 2025) - per-head block sparsity
 3. DuSA dual-stage sparse (NeurIPS 2025) - intra-block dense + inter-block sparse
 4. AnchorAttention stripe granularity (2025) - fine-grained intra-block band sparsity
 5. Spectraformer random Fourier feature rectification (2025) - global linear fallback
 6. Softpick rectified softmax (2025) - natural ~47% activation sparsity

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_nexus(model: torch.nn.modules.module.Module | None = None, layer: torch.nn.modules.module.Module | None = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any) ‑> neuralops.attentions.nexus.NEXUSAttention | torch.nn.modules.module.Module | None`
:   

`inject_nexus_attention(model: torch.nn.modules.module.Module | None = None, layer: torch.nn.modules.module.Module | None = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any) ‑> neuralops.attentions.nexus.NEXUSAttention | torch.nn.modules.module.Module | None`
:   

`softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) ‑> torch.Tensor`
:   

Classes
-------

`DualStageStripeAttention(cfg: neuralops.attentions.nexus.NEXUSConfig, head_dim: int, num_heads: int = 1, is_decoder: bool = False, is_nested_encoder: bool = False, **kwargs)`
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

    `forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, token_valid: torch.Tensor | None = None, seer_mask: torch.Tensor | None = None) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`NEXUSAttention(config: Any | None = None, model_config: Any | None = None, nexus_cfg: neuralops.attentions.nexus.NEXUSConfig | None = None, layer_idx: int = 0, total_layers: int = 6, pretrained_attn: torch.nn.modules.module.Module | None = None, bert_mode: bool = False, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
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

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`NEXUSConfig(block_size: int = 16, top_k_blocks: int = 4, xattn_enabled: bool = True, xattn_stride: int = 2, seer_enabled: bool = True, seer_gate_rank: int = 16, seer_top_k_blocks: int = 4, seer_temperature: float = 1.0, dual_stage_enabled: bool = True, intra_block_full: bool = True, stripe_enabled: bool = True, stripe_width: int = 4, rectify_interval: int = 4, rff_features: int = 128, softpick_enabled: bool = True, softpick_temperature: float = 1.0, sparse_min_seq_len: int = 8, enable_spectraformer: bool = False, dropout: float = 0.1)`
:   NEXUSConfig(block_size: int = 16, top_k_blocks: int = 4, xattn_enabled: bool = True, xattn_stride: int = 2, seer_enabled: bool = True, seer_gate_rank: int = 16, seer_top_k_blocks: int = 4, seer_temperature: float = 1.0, dual_stage_enabled: bool = True, intra_block_full: bool = True, stripe_enabled: bool = True, stripe_width: int = 4, rectify_interval: int = 4, rff_features: int = 128, softpick_enabled: bool = True, softpick_temperature: float = 1.0, sparse_min_seq_len: int = 8, enable_spectraformer: bool = False, dropout: float = 0.1)

    ### Instance variables

    `block_size: int`
    :   The type of the None singleton.

    `dropout: float`
    :   The type of the None singleton.

    `dual_stage_enabled: bool`
    :   The type of the None singleton.

    `enable_spectraformer: bool`
    :   The type of the None singleton.

    `intra_block_full: bool`
    :   The type of the None singleton.

    `rectify_interval: int`
    :   The type of the None singleton.

    `rff_features: int`
    :   The type of the None singleton.

    `seer_enabled: bool`
    :   The type of the None singleton.

    `seer_gate_rank: int`
    :   The type of the None singleton.

    `seer_temperature: float`
    :   The type of the None singleton.

    `seer_top_k_blocks: int`
    :   The type of the None singleton.

    `softpick_enabled: bool`
    :   The type of the None singleton.

    `softpick_temperature: float`
    :   The type of the None singleton.

    `sparse_min_seq_len: int`
    :   The type of the None singleton.

    `stripe_enabled: bool`
    :   The type of the None singleton.

    `stripe_width: int`
    :   The type of the None singleton.

    `top_k_blocks: int`
    :   The type of the None singleton.

    `xattn_enabled: bool`
    :   The type of the None singleton.

    `xattn_stride: int`
    :   The type of the None singleton.

`SeerGating(head_dim: int, num_heads: int, gate_rank: int, top_k: int, temperature: float = 1.0, **kwargs)`
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

    `forward(self, Q: torch.Tensor, K: torch.Tensor, n_blocks: int, training: bool = True) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`SpectraformerRectify(head_dim: int, rff_features: int = 128, **kwargs)`
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

    `forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`XAttentionScorer()`
:   

    ### Static methods

    `score_blocks(Q: torch.Tensor, K: torch.Tensor, block_size: int, stride: int = 2) ‑> torch.Tensor`
    :