Module neuralops.attentions.aegis_spectron_fp8
==============================================
Aegis-Spectron-FP8 Hybrid Attention (Final Corrected).

Three-branch attention with spectral entropy routing:
  1. Local sliding-window attention  (O(N*W))
  2. Global low-rank ELU+1 linear attention (O(N*r*D))
  3. Linear state recurrence (O(N*D), chunked, numerically stable)

Functions
---------

`fp8_cast_simulate(x: torch.Tensor, format: str = 'e4m3', per_channel: bool = False, use_ste: bool | None = None) ‑> torch.Tensor`
:   Simulate FP8 quantization error via dynamic scaling + cast-dequantize roundtrip.
    
    Args:
        x: input tensor (any floating dtype).
        format: "e4m3" (max 448) or "e5m2" (max 57344).
        per_channel: per-last-dim scaling if True, else per-tensor.
        use_ste: If True, forward returns quantized value, gradient flows through
                 the identity (straight-through estimator). If False, forward
                 returns quantized value with no STE. If None, auto: STE in
                 training, plain quantize in eval.

`inject_aegis_spectron_fp8(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`AegisSpectronFP8Attention(config: Any | None = None, hidden_size: int | None = None, num_heads: int | None = None, pretrained_attn: torch.nn.modules.module.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k`
    :

    `k_lin`
    :

    `key`
    :

    `o`
    :

    `o_proj`
    :

    `out_lin`
    :

    `q`
    :

    `q_lin`
    :

    `query`
    :

    `v`
    :

    `v_lin`
    :

    `value`
    :

    ### Methods

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`AegisSpectronFP8Config(**kwargs: Any)`
:   

`FactorizedLinear(in_features: int, out_features: int, rank: int = 128, lora_rank: int = 32, bias: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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

    `is_custom_attention`
    :   The type of the None singleton.

    ### Instance variables

    `bias: torch.Tensor | None`
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

`SpectronSpectralEntropyRouter(config: neuralops.attentions.aegis_spectron_fp8.AegisSpectronFP8Config)`
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

    `forward(self, X: torch.Tensor, token_valid: torch.Tensor | None = None) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.