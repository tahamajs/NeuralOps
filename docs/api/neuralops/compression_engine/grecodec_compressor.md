Module neuralops.compression_engine.grecodec_compressor
=======================================================
GreCoDeC: Greedy Coordinate Descent Low-Rank Compressor.
Registered in COMPRESSOR_REGISTRY as 'grecodec'.

Functions
---------

`grecodec_compressor_factory(model: nn.Module, comp_config: Any | None = None, rank: int | None = None, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   Factory function for GreCoDeC compressor.

`grecodec_factory(model: nn.Module, comp_config: Any | None = None, rank: int | None = None, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   Factory function for GreCoDeC compressor.

Classes
-------

`GreCoDeCCompressor(model: nn.Module | None = None, rank: int = 16, max_iter: int = 15, lora_r: int = 0, lora_alpha: float = 1.0, max_error: float = 0.35, target_attention_only: bool = False, fallback_to_original: bool = True, skip_attention_qkv: bool = False, cfg: Any | None = None, target_layers: list[str] | None = None, **kwargs: Any)`
:   GreCoDeC Compressor.
    Decomposes linear layers using coordinate descent low-rank factorization.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, model: nn.Module | None = None) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress linear layers using GreCoDeC decomposition.

`GreCoDeCLinear(in_features: int, out_features: int, rank: int = 16, bias: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, dtype: torch.dtype | None = None, device: torch.device | None = None)`
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
    :   Greedy Coordinate Descent Factorized Linear Layer with optional LoRA adapter.
        W ≈ U @ V
        - U in R^{out_features x rank}
        - V in R^{rank x in_features}
        
        Forward pass:
        y = ((x @ V^T) @ U^T) + bias + (x @ lora_A^T @ lora_B^T) * (alpha / r)

    ### Static methods

    `from_linear(linear: nn.Linear | Any, rank: int = 16, max_iter: int = 15, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.grecodec_compressor.GreCoDeCLinear`
    :   Construct GreCoDeCLinear by greedy coordinate descent low-rank factorization.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix for compatibility.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Sub-quadratic forward execution.
        y = ((x @ V^T) @ U^T) + bias + LoRA

    `get_weight(self) ‑> torch.Tensor`
    :   Materialize full reconstructed weight matrix W = U @ V.