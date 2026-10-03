Module neuralops.compression_engine.zerokv_neo_compressor
=========================================================
ZeroKV-Neo: L2-norm based KV cache eviction with attention sink and heavy-hitter preservation.
Registered as 'zerokv_neo'.
Production-hardened with:
- Bounded O(1) cache memory management with safe dimension handling
- Relative Frobenius error validation on pruned weights
- Adaptive keep_ratio retry loop
- Conv1D support (GPT-2)
- Critical layer protection (classifier, score, lm_head, pooler, embedding)
- Detailed per-layer error stats logging

Functions
---------

`zerokv_neo_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   ZeroKV-Neo compressor factory function.

Classes
-------

`ZeroKVNeoCache(max_cache_len: int = 128, num_attention_sinks: int = 4, recent_window: int = 32)`
:   Dynamic KV Cache Manager for ZeroKV-Neo.
    Maintains bounded O(1) cache memory by partitioning budget:
    - Pinned attention sinks (first N_sinks prompt tokens)
    - Pinned local sliding window (last N_window recent tokens)
    - Dynamic heavy hitters in the middle sequence retained via L2 norm / cumulative score.

    ### Methods

    `reset(self) ‑> None`
    :   Clear cache tensors to avoid memory leaks.

    `update_and_compress(self, new_k: torch.Tensor, new_v: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
    :   Append new KV tokens and evict non-essential tokens when exceeding capacity.
        Shapes: (B, H, N, D).

`ZeroKVNeoCompressor(model: nn.Module, keep_ratio: float = 0.3, num_attention_sinks: int = 4, heavy_hitter_ratio: float = 0.1, max_error: float = 0.85, max_retries: int = 3, fallback_to_original: bool = True, target_attention_only: bool = True, **kwargs: Any)`
:   ZeroKV-Neo Compressor: L2-norm weight pruning preserving attention sinks and heavy hitters.
    
    Args:
        model: PyTorch model.
        keep_ratio: Fraction of output channels to retain (default 0.3).
        num_attention_sinks: Number of initial sink channels pinned (default 4).
        heavy_hitter_ratio: Fraction allocated to heavy hitters (default 0.1).
        max_error: Maximum allowed relative Frobenius error (default 0.85).
        max_retries: Number of keep_ratio retry steps on error threshold excess (default 3).
        fallback_to_original: If True, keep original layer on high error (default True).
        target_attention_only: If True, only compress attention projections (default True for ZeroKV).

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress linear projection weights preserving attention sinks and heavy hitters.