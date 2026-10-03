Module neuralops.compression_engine.dynamic_rank
================================================
Dynamic Entropy Singular Value Decomposition (Dynamic Entropy SVD) Compressor.

Implements singular entropy computation over empirical layer activation covariances:
$$H(\Sigma_l) = -\sum_{i=1}^{d} p_i \ln(p_i), \quad p_i = \frac{\sigma_i}{\sum_j \sigma_j}$$
Dynamically budgets layer ranks $r_l \in [r_{\min}, r_{\max}]$ proportional to singular entropy,
subject to global rank or parameter budgets, and replaces dense linear layers with low-rank
factorized `DynamicSVDLinear` modules.

Functions
---------

`allocate_entropy_ranks(entropies: dict[str, float], min_rank: int = 16, max_rank: int = 128, total_rank_budget: int | None = None, total_param_budget: int | None = None, target_param_ratio: float | None = None, layer_dims: dict[str, tuple[int, int]] | None = None) ‑> dict[str, int]`
:   Dynamically allocate low-rank budgets r_l in [min_rank, max_rank] across transformer
    layers proportional to singular entropy H(Sigma_l), constrained by total rank or parameter budget.
    
    Args:
        entropies: Dictionary mapping layer names to their singular entropy values.
        min_rank: Lower bound for per-layer rank.
        max_rank: Upper bound for per-layer rank.
        total_rank_budget: Optional total sum of ranks across all layers.
        total_param_budget: Optional total parameter budget for factorized layers.
        target_param_ratio: Optional parameter retention ratio in (0, 1].
        layer_dims: Optional dictionary mapping layer names to (out_features, in_features).
    
    Returns:
        Dict[str, int]: Allocated rank for each layer.

`compute_singular_entropy(spectrum_or_matrix: torch.Tensor, eps: float = 1e-12, normalize: bool = False) ‑> float`
:   Compute Singular Entropy H(Sigma) = -sum(p_i * log(p_i)) of a covariance matrix,
    weight matrix, or 1D singular value spectrum tensor.
    
    Args:
        spectrum_or_matrix: 1D singular values tensor, or 2D covariance/weight matrix.
        eps: Small numerical stabilizer to avoid log(0) and division by zero.
        normalize: If True, normalize entropy by ln(dim) to bound in [0.0, 1.0].
    
    Returns:
        float: Computed singular entropy H(Sigma) >= 0.0.

`dynamic_entropy_inject(model: Any = None, layer: Any = None, **kwargs: Any) ‑> Any`
:   Safe injection hook alias.

`dynamic_entropy_svd_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Dynamic Entropy SVD factory function.

`dynamic_rank_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Dynamic Entropy SVD factory function.

`dynamic_rank_inject(model: Any = None, layer: Any = None, **kwargs: Any) ‑> Any`
:   Safe injection hook alias for dynamic_rank.

`entropy_rank_allocator_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Dynamic Entropy SVD factory function.

`inject_dynamic_entropy(model: Any = None, layer: Any = None, **kwargs: Any) ‑> Any`
:   Safe injection hook for dynamic_entropy. Returns None on failure.

`inject_dynamic_rank(model: Any = None, layer: Any = None, **kwargs: Any) ‑> Any`
:   Safe injection hook alias for dynamic_rank.

Classes
-------

`DynamicEntropySVDCompressor(model: nn.Module, min_rank: int = 16, max_rank: int = 128, target_param_ratio: float | None = None, total_rank_budget: int | None = None, target_rank: int | None = None, max_error: float = 0.35, target_attention_only: bool = False, fallback_to_original: bool = True, calibration_loader: Any = None, calibration_texts: list[str] | None = None, target_modules: list[str] | None = None, tokenizer: Any | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Dynamic Entropy SVD Compressor.
    
    1. Computes singular entropy H(Sigma_l) on layer activation covariances or weights.
    2. Allocates layer ranks within [min_rank, max_rank] subject to parameter/rank budget.
    3. Replaces dense Linear/Conv1D layers with DynamicSVDLinear factorized modules.
    4. Provides relative Frobenius error validation, max_error threshold, and safe fallback.
    5. Freezes fallback and non-target layers during recovery.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Descendants

    * neuralops.compression_engine.dynamic_rank.DynamicEntropySVDCompressorRegistered
    * neuralops.compression_engine.dynamic_rank.EntropyRankAllocatorRegistered

    ### Methods

    `collect_activation_covariances(self) ‑> dict[str, torch.Tensor]`
    :   Collect empirical input activation covariance matrices C_X for each target layer
        using calibration forward passes.

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Execute dynamic entropy-budgeted low-rank SVD compression across all target layers.
        Includes reconstruction error validation, fallback, layer freezing, and parameter tracking.
        
        Returns:
            Tuple[nn.Module, Dict[str, Any]]: Compressed model and detailed statistics.

    `compute_layer_entropies(self, covariances: dict[str, torch.Tensor]) ‑> dict[str, float]`
    :   Compute singular entropy for each target layer using activation covariance
        or fallback weight matrix spectrum.

`DynamicSVDLinear(in_features: int, out_features: int, rank: int, bias: bool = True, dtype: torch.dtype = torch.float32, device: torch.device | None = None)`
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
    :   True Low-Rank Factorized Linear Layer executing as:
        y = (x @ V.T) @ U.T + bias
        where V is (rank, in_features) and U is (out_features, rank).

    ### Static methods

    `from_linear(linear: nn.Module, rank: int, covariance: torch.Tensor | None = None) ‑> neuralops.compression_engine.dynamic_rank.DynamicSVDLinear`
    :   Construct factorized DynamicSVDLinear directly from dense nn.Linear or Conv1D via SVD.
        
        Args:
            linear: Source dense linear or Conv1D layer (or LoRALinear wrapper).
            rank: Target rank for low-rank factorization.
            covariance: Optional empirical activation covariance C_X for activation-aware SVD.
        
        Returns:
            DynamicSVDLinear: Initialized factorized low-rank layer.

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
    :   Forward pass executing factorized low-rank multiplication:
        y = (x @ V^T) @ U^T + bias

    `reset_parameters(self) ‑> None`
    :   Kaiming uniform parameter initialization.