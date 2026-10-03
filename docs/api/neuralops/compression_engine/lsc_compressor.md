Module neuralops.compression_engine.lsc_compressor
==================================================
LSC: Low-Rank Spectral & Fourier-Domain Linear Compressor.
Registered as 'lsc', 'lsc_class', 'lsc_compressor', 'lsc_factory'.

Mathematical Formulation:
  1. 2D DFT: \hat{W} = F_2D(W)
  2. Magnitude: M_ij = |\hat{W}_ij|
  3. Top-k selection: M_spectral = Top-k(M), k = min(r*r, d_out*d_in)
  4. Inverse DFT: W_lsc = Re(F_2D^{-1}(\hat{W} ⊙ M_spectral))

Functions
---------

`lsc_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   LSC Fourier spectral compressor factory.

Classes
-------

`LSCCompressor(model: nn.Module, rank: int = 32, energy_ratio: float | None = None, max_error: float = 0.85, fallback_to_original: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = False, target_attention_only: bool = False, target_modules: list[str] | None = None, min_layer_dim: int = 8, skip_should_compress_filter: bool = False, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`LSCPrunedLoRALinear(in_features: int, out_features: int, weight: torch.Tensor, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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

    `from_linear(linear: nn.Module, weight: torch.Tensor, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.lsc_compressor.LSCPrunedLoRALinear`
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