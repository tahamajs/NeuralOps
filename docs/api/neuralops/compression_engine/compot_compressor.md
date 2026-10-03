Module neuralops.compression_engine.compot_compressor
=====================================================
COMPOT: Compressed Orthogonal Dictionary Learning Compressor.
Registered in COMPRESSOR_REGISTRY as 'compot'. Production hardened for
multi-architecture compatibility (DistilBERT, RoBERTa, GPT-2, Qwen3.5) with
reconstruction validation, Conv1D support, rank reduction bounds, and safe fallback.

Functions
---------

`compot_compressor_factory(model: nn.Module, comp_config: Any | None = None, rank: int | None = None, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   Factory function for COMPOT compression.

Classes
-------

`COMPOTCompressor(model: nn.Module | None = None, rank: int = 64, max_error: float = 0.2, fallback_to_original: bool = True, calibration_texts: list[str] | None = None, calibration_loader: Any | None = None, tokenizer: Any | None = None, cfg: Any | None = None, target_attention_only: bool = False, target_layers: list[str] | None = None, **kwargs: Any)`
:   COMPOT (Compressed Orthogonal Dictionary Learning) Compressor.
    Learns a low-rank basis from activation QR decompositions and compresses
    linear and Conv1D layers into factorized COMPOTLinear modules.
    
    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - Safe fallback to original layer if error exceeds threshold or decomposition is unstable.
    - SVD fallback if QR decomposition on activations fails.
    - Parameter reduction guarantee: checks r * (in + out) < in * out.
    - Robust Conv1D support (GPT-2 c_attn, c_proj) via weight transposition.
    - Unified critical-layer protection via should_compress_linear.
    - Comprehensive per-layer stats and error reporting.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, model: nn.Module | None = None) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress linear and Conv1D layers via COMPOT orthogonal dictionary learning.

`COMPOTLinear(in_features: int, out_features: int, rank: int = 64, bias: bool = True, dtype: torch.dtype | None = None, device: torch.device | None = None)`
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
    :   Factorized Linear Layer constructed via Orthogonal Dictionary Learning (COMPOT):
        W ≈ B @ A
        - A in R^{rank x in_features}: input-orthogonal projection matrix
        - B in R^{out_features x rank}: output dictionary matrix
        
        Forward pass:
        y = ((x @ A^T) @ B^T) + bias

    ### Static methods

    `from_linear_and_activations(layer: nn.Module, activations: torch.Tensor | None = None, rank: int = 64) ‑> tuple[neuralops.compression_engine.compot_compressor.COMPOTLinear, float, bool]`
    :   Construct COMPOTLinear by learning orthogonal dictionary from activations or weights.
        Returns: (compot_layer, relative_reconstruction_error, used_activations)

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix (out_features, in_features).

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Forward pass matching standard Linear and Conv1D.

    `get_weight(self) ‑> torch.Tensor`
    :   Materialize full weight matrix W = B @ A.