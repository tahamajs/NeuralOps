Module neuralops.compression_engine.cur_moe_compressor
======================================================
CUR-MoE Compressor: apply CUR cross-approximation to Mixture of Experts (MoE) modules and expert layers.
Registered in COMPRESSOR_REGISTRY as 'cur_moe' and 'cur_moe_compressor'.

Functions
---------

`cur_moe_factory(model: nn.Module, comp_config: Any | None = None, rank: int | None = None, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   Factory function for CUR-MoE compressor.

`cur_moe_inject(model: nn.Module, **kwargs: Any) ‑> torch.nn.modules.module.Module | None`
:   Safe injection hook for cur_moe alias.

`inject_cur_moe(model: nn.Module, **kwargs: Any) ‑> torch.nn.modules.module.Module | None`
:   Safe injection hook for cur_moe. Catches exceptions, logs, and returns None on failure.

Classes
-------

`CURMoECompressor(model: nn.Module | None = None, rank: int = 64, min_rank: int = 8, method: str = 'leverage', deterministic: bool = True, max_error: float = 0.9, fallback_to_original: bool = True, max_retries: int = 2, target_attention_only: bool = False, cfg: Any | None = None, **kwargs: Any)`
:   Compresses Mixture of Experts (MoE) layers and dense FFN layers using CUR cross-approximation.
    
    Args:
        model: PyTorch model containing MoE or transformer layers.
        rank: Target rank for CUR decomposition.
        min_rank: Minimum allowable rank (default 8).
        method: 'leverage' (statistical leverage scores) or 'energy' (Frobenius energy).
        deterministic: Whether to use deterministic top-k selection (default True).
        max_error: Maximum allowed relative Frobenius error (default 0.90).
        fallback_to_original: Whether to preserve original layer if error > max_error (default True).
        max_retries: Number of adaptive rank escalations (default 2).
        target_attention_only: If True, skip attention projections and compress only FFN/MoE expert layers.
        cfg: Optional configuration object.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, model: nn.Module | None = None) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress expert linear layers in MoE architecture, or fallback to FFN layers.