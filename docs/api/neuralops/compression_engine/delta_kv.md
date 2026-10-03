Module neuralops.compression_engine.delta_kv
============================================
DeltaKV: Residual-Based KV Cache Compression via Long-Range Similarity.
Registered as 'delta_kv', 'delta_kv_class', 'delta_kv_compressor', 'delta_kv_factory'.

Mathematical formulation:
  W ≈ L + R
  L = U_r Σ_r V_r^T                     (low-rank, rank = r)
  R = sparse(|W - L|; 1 - sparsity)     (top-k residual magnitudes)
  W_compressed = L + R

Functions
---------

`delta_kv_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   DeltaKV compressor factory.

Classes
-------

`DeltaKVCompressor(model: nn.Module, rank: int = 16, sparsity: float = 0.29, max_error: float = 0.35, target_attention_only: bool = False, fallback_to_original: bool = True, min_layer_dim: int = 8, skip_should_compress_filter: bool = False, cfg: Any | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`DeltaKVStore(max_refs: int = 128, topk: int = 4, compression_ratio: float = 0.29, residual_threshold: float = 0.001)`
:   Optional runtime cache that stores delta residuals relative to retrieved
    historical K/V references. Not used by the weight-only compressor path.

    ### Methods

    `compress(self, k: torch.Tensor, v: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
    :

    `reset(self) ‑> None`
    :

    `update(self, k: torch.Tensor, v: torch.Tensor) ‑> None`
    :