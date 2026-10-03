Module neuralops.compression_engine.dobi_svd
============================================
Dobi-SVD: Differentiable SVD layer with learnable spectral gating.
Registered as 'dobi' and 'dobi_svd'. Production hardened for multi-architecture
compatibility (DistilBERT, RoBERTa, GPT-2, Qwen3.5) with error validation, Conv1D support,
and safe fallback.

Functions
---------

`dobi_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Dobi factory function.

`dobi_inject(model: Any = None, layer: Any = None, **kwargs: Any) ‑> Any`
:   Safe injection hook alias for dobi.

`dobi_svd_compressor_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Dobi factory function.

`dobi_svd_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Dobi factory function.

`inject_dobi(model: Any = None, layer: Any = None, **kwargs: Any) ‑> Any`
:   Safe injection hook for Dobi-SVD. Returns None on failure.

Classes
-------

`DobiSVDCompressor(model: nn.Module, init_rank_ratio: float = 0.5, rank: int | None = None, init_gate_val: float = 2.5, max_error: float = 0.35, fallback_to_original: bool = True, target_attention_only: bool = True, max_retries: int = 2, target_modules: list[str] | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Dobi-SVD Compressor: Differentiable SVD with learnable spectral gating.
    
    Extracts truncated SVD factors from pretrained weights with symmetric
    square-root singular value distribution (U_r * sqrt(Sigma_r), sqrt(Sigma_r) * V_r^T),
    compensates for initial gate value to ensure near-lossless initialization, and
    replaces layers with DobiSVDLinearLayer containing learnable gating parameters.
    
    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - Automatic rank escalation retries when error exceeds threshold.
    - Safe fallback to original module if error remains high.
    - Fallback and non-target layers frozen (requires_grad=False) during recovery.
    - Robust LoRALinear and Conv1D support (with transpose and adapter preservation).
    - target_attention_only parameter: restricts SVD factorization to attention projections.
    - Strict rank bounds guaranteeing parameter reduction.
    - Critical-layer protection via should_compress_linear.
    - Detailed per-layer stats, avg error, and effective parameter ratio.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress all target linear and Conv1D layers across the model.

    `compress_linear(self, layer: nn.Module, rank: int | None = None) ‑> tuple[neuralops.compression_engine.dobi_svd.DobiSVDLinearLayer | None, float, int]`
    :   Decompose a single linear or Conv1D layer into DobiSVDLinearLayer.

`DobiSVDLinearLayer(in_features: int, out_features: int, rank: int, bias: bool = True, init_gate_val: float = 2.5, dtype: torch.dtype = torch.float32, device: torch.device | None = None)`
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
    :   Differentiable SVD Linear Layer (Dobi-SVD).
        
        Decomposes linear transformations into low-rank factorized parameters
        (U: [out_features, rank], V: [rank, in_features]) with a learnable continuous
        spectral gate vector alpha: [rank], allowing differentiable rank pruning and fine-tuning:
            y = (x @ (V * sigmoid(alpha)).T) @ U.T + bias

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix for compatibility.

    ### Methods

    `extra_repr(self) ‑> str`
    :   Return the extra representation of the module.
        
        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Differentiable forward pass with spectral gating matching Linear and Conv1D.

    `reset_parameters(self) ‑> None`
    :

    `set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) ‑> None`
    :   Initialize U and V parameters directly from SVD factors.