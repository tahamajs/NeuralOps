Module neuralops.compression_engine.oats_compressor
===================================================
OATS: Outlier-Aware Training & Saliency-Based Sparsification with Low-Rank Matrix Decomposition.
Registered in COMPRESSOR_REGISTRY as 'oats' and 'oats_compressor'.

Mathematical Formulation:
1. Activation Second Moment:
   $$M = \frac{1}{N} X^\top X + \epsilon I \in \mathbb{R}^{d_{in} \times d_{in}}$$
2. Cholesky Factorization:
   $$M = L_M L_M^\top$$
3. Preconditioned Weight Matrix:
   $$\widetilde{W} = W L_M$$
4. Alternating Low-Rank + Sparse Decomposition:
   - Low-Rank Outlier Component: $L = \operatorname{SVD}_r(\widetilde{W} - S)$
   - Sparse Residual Component: $S = \Pi_{2:4}(\widetilde{W} - L)$ or $\operatorname{Top-}(1-s)(\widetilde{W} - L)$
5. De-conditioned Weight Matrix:
   $$W_{oats} = (L + S) L_M^{-1}$$

Functions
---------

`oats_compressor_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Factory function for OATS compressor.

Classes
-------

`OATSCompressor(model: nn.Module, tokenizer: Any | None = None, sparsity: float = 0.5, rank_ratio: float = 0.2, num_iters: int = 4, device: str | torch.device | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, structured_24: bool = True, max_error: float = 0.85, fallback_to_original: bool = True, skip_attention_qkv: bool = False, target_attention_only: bool = False, target_modules: list[str] | None = None, lora_r: int = 0, lora_alpha: float = 1.0, min_rank: int = 4, **kwargs: Any)`
:   OATS Compressor: Outlier-aware low-rank + structured sparse matrix decomposition.
    
    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error (default 0.85).
    - Safe fallback to original weights by default (fallback_to_original=True).
    - Robust Conv1D detection by class name and nf attribute without unreliable shape heuristics.
    - Unified critical-layer protection via should_compress_linear.
    - GQA-aware skipping to protect asymmetric attention heads.
    - Tied / shared weight protection tracking parameter IDs.
    - Trainable LoRA adapter integration with base weights frozen.
    - FP32 numerical stability with NaN/Inf guards.
    - Post-compression forward sanity check under torch.no_grad().
    - Comprehensive per-layer stats and error tracking.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Executes OATS decomposition across compressible Linear and Conv1D modules.

`OATSLoRALinear(in_features: int, out_features: int, weight: torch.Tensor, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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
    :   Linear layer containing OATS decomposed base weights (frozen)
        with optional trainable LoRA adapters (lora_A, lora_B).

    ### Static methods

    `from_linear(linear: nn.Module, weight: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, **kwargs: Any) ‑> neuralops.compression_engine.oats_compressor.OATSLoRALinear`
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