Module neuralops.compression_engine.svd_linear
==============================================
SVD-based low-rank linear layer replacement with optional Fused SVD kernel acceleration.
Registered as 'svd_linear' and 'svd_linear_compressor'.

Functions
---------

`fused_svd_linear(x: torch.Tensor, V: torch.Tensor, U: torch.Tensor, norm_weight: torch.Tensor | None = None, bias: torch.Tensor | None = None, eps: float = 1e-06, use_rms_norm: bool = False) ‑> torch.Tensor`
:   Functional interface for Fused SVD-Linear with optional in-SRAM RMSNorm.
    
    Args:
        x: Input tensor of shape (..., in_features).
        V: Down-projection matrix of shape (rank, in_features).
        U: Up-projection matrix of shape (out_features, rank).
        norm_weight: Optional RMSNorm affine scale vector of shape (rank,).
        bias: Optional bias vector of shape (out_features,).
        eps: Small epsilon for RMSNorm numerical stability.
        use_rms_norm: Whether to apply in-SRAM RMSNorm on intermediate rank activations.
    
    Returns:
        Output tensor of shape (..., out_features).

`svd_linear_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   SVD Linear compressor factory.

Classes
-------

`FusedSVDLinear(in_features: int, out_features: int, rank: int = 32, bias: bool = True, use_rms_norm: bool = False, eps: float = 1e-06, dtype: torch.dtype = torch.float32, device: torch.device | None = None)`
:   RESEARCH SOTA: Fused SVD-Linear Module.
    Combines low-rank projection $Z = x V^T$, in-SRAM RMSNorm, and up-projection $Y = Z U^T + b$
    into a single kernel invocation, eliminating HBM roundtrips.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix for compatibility.

    ### Methods

    `decompose_from_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) ‑> None`
    :   Decompose dense weight matrix into symmetric SVD factors.

    `extra_repr(self) ‑> str`
    :   Return the extra representation of the module.
        
        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `get_orthogonality_loss(self) ‑> torch.Tensor`
    :   # RESEARCH FIX: Computes Stiefel manifold penalty to prevent singular subspace collapse.

    `reset_parameters(self) ‑> None`
    :

    `set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) ‑> None`
    :   Set precomputed U and V low-rank factors.

`SVDLinearCompressor(model: nn.Module, rank: int = 192, max_error: float = 0.2, fallback_to_original: bool = True, protect_critical: bool = True, target_attention_only: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = False, target_modules: list[str] | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Compresses model by replacing Linear/Conv1D layers with SVDLinearLayer.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress linear layers with SVD factorization.

`SVDLinearLayer(linear_layer: nn.Module, rank: int, use_fused_kernel: bool = False, use_rms_norm: bool = False, eps: float = 1e-06, lora_r: int = 0, lora_alpha: float = 1.0)`
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
    :   Replace a linear layer with factorized U @ V, keeping top-`rank` components.
        Optionally dispatches fused in-SRAM kernel to eliminate memory roundtrips.
        Includes optional trainable LoRA adapters (lora_A, lora_B).

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix for compatibility.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Forward pass with optional fused kernel acceleration and LoRA branch.