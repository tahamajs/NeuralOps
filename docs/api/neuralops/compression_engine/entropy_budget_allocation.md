Module neuralops.compression_engine.entropy_budget_allocation
=============================================================
EntropyBudgetAllocation: global rank assignment via whitened activation entropy.
Registered as 'entropy_budget' and 'entropy_budget_allocation'.

Functions
---------

`entropy_budget_allocation_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> torch.nn.modules.module.Module`
:   Entropy budget factory function.

`entropy_budget_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> torch.nn.modules.module.Module`
:   Entropy budget factory function.

Classes
-------

`EntropyBudgetCompressor(model: nn.Module, calibration_loader: Any | None = None, total_budget_ratio: float = 0.7, rank: int | None = None, eps: float = 1e-06, max_error: float = 0.35, target_attention_only: bool = False, fallback_to_original: bool = True, target_modules: list[str] | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   Entropy Budget Low-Rank SVD Compressor.
        
        Computes activation covariance and whiten weights, calculates per-component
        singular entropy contributions, allocates low-rank budgets globally across layers,
        and replaces dense nn.Linear layers with factorized SVDLinear modules with symmetric
        singular value splitting (U * sqrt(Sigma), sqrt(Sigma) * V^T).
        
        Hardened for:
        - LoRALinear wrappers & Conv1D support.
        - target_attention_only filtering.
        - Relative Frobenius reconstruction error validation & fallback.
        - Freezing of fallback and skipped layers.
        - Real compression metrics recording.

    ### Methods

    `collect_covariances(self) ‑> dict[str, torch.Tensor]`
    :   Collect empirical input activation covariances with double-precision accumulation.

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress model using whitened entropy budget allocation and covariance-aligned singular vectors.