Module neuralops.compression_engine.aatc_xkv_compressor
=======================================================
AATC-XKV: Attention-Aware Cross-Layer KV Cache Compression (Production-Ready v2).

Real KV cache compression:
  1. AATC sensitivity-weighted bit allocation (reverse waterfilling).
  2. xKV cross-layer shared subspace via joint SVD on REAL calibration data.
  3. DynamicCache subclass that stores int8 quantized coefficients instead of fp16 K/V.
  4. Attention sink preservation (first token) + optional last-token preservation
     in PREFILL only.

Functions
---------

`dequantize_coefficients(q: torch.Tensor, scales: torch.Tensor) ‑> torch.Tensor`
:   

`quantize_coefficients(U: torch.Tensor, bit_alloc: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Quantize U along the last dimension with per-component bit widths.
    Components with b ≤ 0 are stored as zero with unit scale.

`reverse_waterfilling(sigma_sq: torch.Tensor, sensitivity: torch.Tensor, total_bits: int, min_bits: int = 0, max_bits: int = 8, max_iter: int = 100) ‑> torch.Tensor`
:   Distribute `total_bits` across r components to minimize
        Σ s_c · σ_c² · 2^(-2 b_c)
    subject to Σ b_c ≤ total_bits and min_bits ≤ b_c ≤ max_bits.
    
    Uses a geometric-mean binary search on λ in log-space:
        b_c = clamp(0.5 · log2(w_c / λ), min_bits, max_bits)
    where w_c = s_c · σ_c².

Classes
-------

`AATC_XKVCache(subspaces: dict[int, SharedSubspace], layer_to_group: dict[int, int], sink_tokens: int = 1, preserve_last_token: bool = True)`
:   Drop-in replacement for HF DynamicCache that transparently compresses K/V
    via AATC-XKV.
    
    Storage per (layer, batch, head, token):
      - q_k: int8 coefficients (r_k,) + fp32 scale vector (r_k,)
      - q_v: int8 coefficients (r_v,) + fp32 scale vector (r_v,)
      - sink_k/sink_v (fp32) for the first `sink_tokens` tokens of the sequence
      - last_k/last_v (fp32) only during PREFILL (T_new > 1) when
        `preserve_last_token=True`
    
    Reconstruction:
      - `_reconstruction_cache[layer_idx]` holds the fully-reconstructed
        fp16 K/V up to the last decompressed entry, so decoding only pays
        O(new_tokens · r · D) per step.

    ### Ancestors (in MRO)

    * transformers.cache_utils.DynamicCache
    * transformers.cache_utils.Cache

    ### Methods

    `get_max_cache_shape(self) ‑> int`
    :   Returns maximum sequence length of the cache object. Dynamic caches do not have a maximum length.

    `get_max_length(self) ‑> int | None`
    :

    `get_seq_length(self, layer_idx: int = 0) ‑> int`
    :   Returns the sequence length of the cache for the given layer.

    `get_usable_length(self, new_seq_length: int, layer_idx: int = 0) ‑> int`
    :

    `memory_bytes(self) ‑> dict[str, int]`
    :

    `memory_savings_ratio(self) ‑> float`
    :

    `reorder_cache(self, beam_idx: torch.LongTensor) ‑> None`
    :   AXKV-FIX-05: correct beam-search reorder. Rebuilds the reconstruction
        cache lazily after reordering the compressed entries.

    `reset(self) ‑> None`
    :   Recursively reset all layers tensors

    `update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs: dict[str, Any] | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
    :   Append new K/V, return the full K/V needed for attention.

`AATC_XKVCompressor(model: nn.Module, group_size: int = 3, target_bits_per_component: float = 4.0, energy_threshold: float = 0.95, sink_tokens: int = 1, preserve_last_token: bool = True, calib_batches: list[Any] | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `IS_DYNAMIC_KV_CACHE`
    :   The type of the None singleton.

    `is_custom_attention: bool`
    :   The type of the None singleton.

`SharedSubspace(V_k: torch.Tensor, V_v: torch.Tensor, mean_k: torch.Tensor, mean_v: torch.Tensor, bit_alloc_k: torch.Tensor, bit_alloc_v: torch.Tensor, group_id: int = 0, layer_indices: list[int] = <factory>)`
:   Shared low-rank subspace for a group of consecutive transformer layers.

    ### Instance variables

    `V_k: torch.Tensor`
    :   The type of the None singleton.

    `V_v: torch.Tensor`
    :   The type of the None singleton.

    `bit_alloc_k: torch.Tensor`
    :   The type of the None singleton.

    `bit_alloc_v: torch.Tensor`
    :   The type of the None singleton.

    `group_id: int`
    :   The type of the None singleton.

    `layer_indices: list[int]`
    :   The type of the None singleton.

    `mean_k: torch.Tensor`
    :   The type of the None singleton.

    `mean_v: torch.Tensor`
    :   The type of the None singleton.