Module neuralops.compression_engine.kronecker_tt_lora
=====================================================
Kronecker-TT-LoRA Hybrid Compression Engine Adapter.

Decomposes weight updates as:
  ΔW = γ * ((A1 ⊗ A2) ⊙ TT(G1, G2, G3))[:out, :in] + (α/r) * (B_lora @ A_lora)

where (A1 ⊗ A2) is a Kronecker product and TT is a 3-core Tensor-Train.

Functions
---------

`kronecker_tt_lora_factory(model: torch.nn.modules.module.Module, comp_config: Any | None = None, rank: int | None = None, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   Factory for Kronecker-TT-LoRA compressor.

Classes
-------

`KroneckerTTLoRACompressor(model: torch.nn.modules.module.Module | None = None, rank: int = 8, gamma: float = 1.0, lora_r: int = 4, lora_alpha: float = 8.0, max_error: float = 0.35, target_attention_only: bool = False, fallback_to_original: bool = True, skip_attention_qkv: bool = False, skip_should_compress_filter: bool = False, min_layer_dim: int = 8, cfg: Any | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress_linear(self, linear_layer: torch.nn.modules.module.Module) ‑> neuralops.compression_engine.kronecker_tt_lora.KroneckerTTLoRALayer`
    :

`KroneckerTTLoRALayer(base_layer: torch.nn.modules.linear.Linear, m1: int, n1: int, m2: int, n2: int, tt_shapes: tuple[int, int, int], tt_ranks: tuple[int, int, int, int] = (1, 4, 4, 1), gamma: float = 1.0, lora_r: int = 4, lora_alpha: float = 8.0, r_min: int = 2, r_max: int = 32, entropy_threshold: float = 0.75)`
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

    `is_custom_linear: bool`
    :   The type of the None singleton.

    ### Static methods

    `factorize_dimensions(in_f: int, out_f: int) ‑> dict[str, typing.Any]`
    :   Auto-factorize dims and pick 3-factor TT shape.

    `from_linear(base_layer: torch.nn.modules.module.Module, m1: int | None = None, n1: int | None = None, m2: int | None = None, n2: int | None = None, tt_shapes: tuple[int, int, int] | None = None, tt_ranks: tuple[int, int, int, int] = (1, 4, 4, 1), gamma: float = 1.0, lora_r: int = 4, lora_alpha: float = 8.0, **kwargs: Any) ‑> neuralops.compression_engine.kronecker_tt_lora.KroneckerTTLoRALayer`
    :   Build a KroneckerTTLoRALayer from Linear / Conv1D / LoRA wrapper.

    ### Instance variables

    `bias: torch.Tensor | None`
    :

    `weight: torch.Tensor`
    :

    ### Methods

    `adapt_tt_ranks(self, delta_r: int = 1) ‑> None`
    :

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `get_delta_w(self) ‑> torch.Tensor`
    :

    `get_tt_tensor(self) ‑> torch.Tensor`
    :