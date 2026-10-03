Module neuralops.compression_engine.tt_compressor
=================================================
Tensor Train Compressor (best-effort, production-guarded).

Registered as 'tt', 'tt_linear', 'tt_compressor', 'tt_linear_compressor',
              'tt_factory', 'tt_compressor_factory', 'tt_linear_factory'.

⚠️  FUNDAMENTAL LIMITATION ⚠️

TT decomposition assumes a matrix W reshaped as a multi-dimensional tensor
(q_0, q_1, ..., p_0, p_1, ...) admits a low-rank TT factorization. Pretrained
transformer weights — especially fused Q/K/V projections in GPT-2 — do NOT
exhibit this structure. Naive TT therefore destroys the model.

Empirical failures with the previous implementation:
  - RoBERTa-base SST-2 : 50.92%  (chance baseline 50.9%, original 94.5%)
  - GPT-2 WikiText-2   : PPL 2156 (baseline 41.35)
  - DistilBERT SST-2   : ~58%    (baseline 90.71%)

This hardened version mitigates the failure by:
  1. Using adaptive TT-SVD (rank chosen per core, based on energy, not fixed).
  2. Adding a per-layer **SVD fallback** when TT error exceeds max_error.
     Only layers where TT succeeds are replaced with TTLinearLayer; the rest
     fall back to a low-rank SVD layer or to the original (per policy).
  3. Conservative dimension factorization that avoids pathological splits.
  4. Full FP32 computation, deterministic caching, and honest stats.

Functions
---------

`tt_compressor_factory(model: nn.Module, comp_config: Any | None = None, rank: int | None = None, **kwargs: Any) ‑> Any`
:   TT compressor factory. Falls back to SVD per-layer if TT fails.

`tt_linear_compressor_factory(model: nn.Module, comp_config: Any | None = None, rank: int | None = None, **kwargs: Any) ‑> Any`
:   TT compressor factory. Falls back to SVD per-layer if TT fails.

Classes
-------

`SVDLinearLayer(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor | None = None, is_conv1d: bool = False)`
:   Low-rank SVD fallback used when TT cannot achieve the error threshold.
    
    Stores A (out, r) and B (r, in) with bias. Forward: y = (x @ B.T) @ A.T + bias.
    Parameter count: r * (out + in) + out (bias).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

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

`TTCompressor(model: nn.Module | None = None, rank: int | None = None, tt_rank: int = 16, num_factors: int = 2, cfg: Any | None = None, target_layers: list[str] | None = None, max_error: float = 0.2, energy_threshold: float = 0.995, fallback_policy: str = 'svd', target_attention_only: bool = False, strict_rank: bool = False, **kwargs: Any)`
:   Replaces Linear/Conv1D with TTLinearLayer when TT succeeds; otherwise
    falls back to SVDLinearLayer. Layers that neither can compress within
    error budget remain untouched.
    
    Args:
        tt_rank:           TT rank bound.
        num_factors:       Number of TT factors per dimension (default 2).
        max_error:         Relative Frobenius error threshold (default 0.20).
        energy_threshold:  Energy retained per TT core (default 0.995).
        fallback_policy:   'svd' | 'original' | 'none'. Default: 'svd'.
        target_attention_only: Only compress attention projections.
        strict_rank:       Do not exceed tt_rank.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress_linear(self, linear_layer: nn.Module) ‑> torch.nn.modules.module.Module | None`
    :   Compress a single layer to TT (or SVD if TT fails).

`TTLinearLayer(in_features: int, out_features: int, tt_rank: int = 16, bias: bool = True, shape: tuple[tuple[int, ...], tuple[int, ...]] | None = None, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Tensor-Train (TT) linear layer.
    
    Weight W ∈ R^{out × in} is reshaped to a multi-dim tensor of shape
    (q_0, q_1, ..., p_0, p_1, ...) with q_dims × p_dims = out × in, then
    factorized into TT cores G_1 ... G_d of shape (r_{k-1}, p_k, q_k, r_k),
    with boundary ranks r_0 = r_d = 1.
    
    Forward contracts TT cores with x directly, without materializing W
    for d == 2; for larger d, an optimized generic path is used.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_linear(linear_layer: nn.Module, tt_rank: int = 16, num_factors: int = 2, max_error: float = 0.2, max_retries: int = 3, energy_threshold: float = 0.995, strict_rank: bool = False) ‑> neuralops.compression_engine.tt_compressor.TTLinearLayer | None`
    :   Factorize a pretrained Linear/Conv1D into TT-cores with adaptive rank.
        
        Rank per core is chosen adaptively: the smallest r such that the
        retained energy ≥ energy_threshold, capped by tt_rank and by the
        max_rank_for_compression budget.
        
        Returns None if the layer cannot be compressed within the error
        threshold and parameter budget.

    ### Instance variables

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

    `materialize_weight(self) ‑> torch.Tensor`
    :   Reconstruct W ∈ R^{out × in} by contracting TT-cores.

`TTLinear(in_features: int, out_features: int, tt_rank: int = 16, bias: bool = True, shape: tuple[tuple[int, ...], tuple[int, ...]] | None = None, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Tensor-Train (TT) linear layer.
    
    Weight W ∈ R^{out × in} is reshaped to a multi-dim tensor of shape
    (q_0, q_1, ..., p_0, p_1, ...) with q_dims × p_dims = out × in, then
    factorized into TT cores G_1 ... G_d of shape (r_{k-1}, p_k, q_k, r_k),
    with boundary ranks r_0 = r_d = 1.
    
    Forward contracts TT cores with x directly, without materializing W
    for d == 2; for larger d, an optimized generic path is used.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_linear(linear_layer: nn.Module, tt_rank: int = 16, num_factors: int = 2, max_error: float = 0.2, max_retries: int = 3, energy_threshold: float = 0.995, strict_rank: bool = False) ‑> neuralops.compression_engine.tt_compressor.TTLinearLayer | None`
    :   Factorize a pretrained Linear/Conv1D into TT-cores with adaptive rank.
        
        Rank per core is chosen adaptively: the smallest r such that the
        retained energy ≥ energy_threshold, capped by tt_rank and by the
        max_rank_for_compression budget.
        
        Returns None if the layer cannot be compressed within the error
        threshold and parameter budget.

    ### Instance variables

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

    `materialize_weight(self) ‑> torch.Tensor`
    :   Reconstruct W ∈ R^{out × in} by contracting TT-cores.