Module neuralops.compression_engine.hybridkv_compressor
=======================================================
HybridKV: Hybrid KV Cache Compression for Multimodal & Long-Context LLMs.
Classifies attention heads into static/dynamic types and applies distinct
compression strategies per type under a hierarchical budget strictly to dynamic KV caches.

Paper: "HybridKV: Hybrid KV Cache Compression for Efficient MLLM Inference"
       Zeng et al., arXiv:2604.05887.
Registered as 'hybridkv'.

Functions
---------

`hybridkv_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   HybridKV compressor factory function.

Classes
-------

`HybridKVCache(max_cache_len: int = 128, num_sinks: int = 4, recent_window: int = 32, budget: float = 0.3, static_ratio: float = 0.7, static_heads_mask: torch.Tensor | None = None)`
:   Dynamic KV Cache Manager for HybridKV.
    Maintains separate retention policies for static vs dynamic attention heads:
    - Static heads: pinned attention sinks (first S tokens) + global heavy hitters
    - Dynamic heads: recent local sliding window (last W tokens) + chunked key retention
    Preserves all model stationary weights and strictly compresses dynamic KV cache tensors.
    Supports variable sequence lengths across prefill and token-by-token decoding.

    ### Methods

    `reset(self) ‑> None`
    :   Clear cache state to prevent memory leaks.

    `update_and_compress(self, new_k: torch.Tensor, new_v: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
    :   Append new KV representations and apply hybrid eviction if cache exceeds capacity.
        Shapes: (B, H, N, D) or (B, N, D) or (N, D).

`HybridKVCompressor(model: nn.Module, budget: float = 0.3, static_ratio: float = 0.7, max_cache_len: int = 128, num_sinks: int = 4, recent_window: int = 32, target_attention_only: bool = True, cfg: Any | None = None, **kwargs: Any)`
:   HybridKV Compressor: Classifies projection heads into static/dynamic types
    and attaches HybridKVCache instances to attention layers without modifying model weights.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Classify heads and configure dynamic KV caches without mutating model weights.