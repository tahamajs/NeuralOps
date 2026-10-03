Module neuralops.compression_engine.hashed_net_compressor
=========================================================
HashedNet Compressor — Production-Hardened (v2).

Registered as 'hashed_net', 'hashednets', 'hashednets_compressor',
              'hashed_linear', 'hashed_net_factory', ...

⚠️  FUNDAMENTAL LIMITATION ⚠️

HashedNet (Chen et al., 2015) compresses a weight matrix W ∈ R^{out×in} by
storing a compact parameter table w ∈ R^{n_buckets} and reconstructing
W[i,j] = sign(i,j) · w[h(i,j)]. This is lossy: intra-bucket variance is
discarded. For random Gaussian weight matrices, relative Frobenius error is
approximately sqrt(1 − 1/avg_bucket_size). For avg_bucket_size = 9 the error
is ~94%. Pretrained transformer weights are NOT an exception — they do NOT
exhibit the structured correlation that would make this compression viable.

Empirically, HashedNet in a compression setting either:
  (a) falls back to the original layer for most layers when max_error is
      strict, or
  (b) produces a heavily degraded model when max_error is relaxed.

This implementation is correct but should be treated as a research baseline,
not a production compressor. For real compression, use SVD (UniQL), MLA, or
structured pruning (ARMOR, FLAT-LLM).

Functions
---------

`fast_hash_and_sign(out_features: int, in_features: int, n_buckets: int, device: torch.device | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Vectorized hash and sign for every (i, j) with 0 ≤ i < out_features,
    0 ≤ j < in_features.
    
    Uses splitmix-style bit mixing that stays within int64 bounds.
    Returns:
        hash_idx: (out, in) int32
        sign_val: (out, in) int8

`hashed_net_factory(model: nn.Module, comp_config: Any | None = None, rank: int | float | None = None, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   

`hashednets_factory(model: nn.Module, comp_config: Any | None = None, rank: int | float | None = None, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   

Classes
-------

`HashedLinear(in_features: int, out_features: int, n_buckets: int = 65536, bias: bool = True, use_sign: bool = True, lora_rank: int = 0, lora_alpha: float = 1.0, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Linear layer with parameter sharing via a compact hash table.
    
      W[i, j] = sign[i, j] · weight_table[hash_idx[i, j]]
      y = x @ W^T + bias + (alpha/r) · (x @ lora_A^T) @ lora_B^T
    
    Storage:
      weight_table:   n_buckets · dtype_size        (the only trained/learned params)
      bias:           out_features · dtype_size
      lora_A, lora_B: (optional) r · (in + out) · dtype_size
      hash_idx:       out · in · 4 bytes  (int32, non-persistent)
      sign_val:       out · in · 1 byte   (int8, non-persistent)
    
    For compression to actually reduce storage, `out · in · 5` bytes must be
    less than `out · in · dtype_size`. With fp16 (2 bytes) this is almost
    never true unless a sparse storage format is used. Honest accounting is
    therefore critical (see HN-FIX-08).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_linear(layer: nn.Module, n_buckets: int = 65536, use_sign: bool = True, lora_rank: int = 0, lora_alpha: float = 1.0, fit_lora_residual: bool = True, **_ignored: Any) ‑> neuralops.compression_engine.hashed_net_compressor.HashedLinear`
    :   Build a HashedLinear by projecting W onto the hash basis.
        
        Reconstruction: W_hat[i,j] = sign[i,j] · mean_{bucket}(W · sign)
        which is the least-squares optimum under the HashedNet constraint.
        
        HN-FIX-07: no energy rescaling (it increases error).

    ### Instance variables

    `bucket_indices: torch.Tensor`
    :

    `weight: torch.Tensor`
    :

    ### Methods

    `extra_repr(self) ‑> str`
    :   Return the extra representation of the module.
        
        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `get_weight(self) ‑> torch.Tensor`
    :

    `storage_bytes(self) ‑> dict[str, int]`
    :   HN-FIX-08: report per-component storage (bytes).
        Used to compute honest compression ratios.

`HashedLinearLayer(in_features: int, out_features: int, n_buckets: int = 65536, bias: bool = True, use_sign: bool = True, lora_rank: int = 0, lora_alpha: float = 1.0, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Linear layer with parameter sharing via a compact hash table.
    
      W[i, j] = sign[i, j] · weight_table[hash_idx[i, j]]
      y = x @ W^T + bias + (alpha/r) · (x @ lora_A^T) @ lora_B^T
    
    Storage:
      weight_table:   n_buckets · dtype_size        (the only trained/learned params)
      bias:           out_features · dtype_size
      lora_A, lora_B: (optional) r · (in + out) · dtype_size
      hash_idx:       out · in · 4 bytes  (int32, non-persistent)
      sign_val:       out · in · 1 byte   (int8, non-persistent)
    
    For compression to actually reduce storage, `out · in · 5` bytes must be
    less than `out · in · dtype_size`. With fp16 (2 bytes) this is almost
    never true unless a sparse storage format is used. Honest accounting is
    therefore critical (see HN-FIX-08).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_linear(layer: nn.Module, n_buckets: int = 65536, use_sign: bool = True, lora_rank: int = 0, lora_alpha: float = 1.0, fit_lora_residual: bool = True, **_ignored: Any) ‑> neuralops.compression_engine.hashed_net_compressor.HashedLinear`
    :   Build a HashedLinear by projecting W onto the hash basis.
        
        Reconstruction: W_hat[i,j] = sign[i,j] · mean_{bucket}(W · sign)
        which is the least-squares optimum under the HashedNet constraint.
        
        HN-FIX-07: no energy rescaling (it increases error).

    ### Instance variables

    `bucket_indices: torch.Tensor`
    :

    `weight: torch.Tensor`
    :

    ### Methods

    `extra_repr(self) ‑> str`
    :   Return the extra representation of the module.
        
        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `get_weight(self) ‑> torch.Tensor`
    :

    `storage_bytes(self) ‑> dict[str, int]`
    :   HN-FIX-08: report per-component storage (bytes).
        Used to compute honest compression ratios.

`HashedNetCompressor(model: nn.Module | None = None, n_buckets: int | None = None, compression_ratio: float | None = None, rank: int | float | None = None, max_error: float = 0.35, target_attention_only: bool = False, fallback_to_original: bool = True, max_retries: int = 2, use_sign: bool = True, lora_rank: int = 0, lora_alpha: float = 1.0, freeze_hash_table: bool = False, skip_attention_qkv: bool = False, cfg: Any | None = None, target_layers: list[str] | None = None, **kwargs: Any)`
:   Replace target linear/Conv1D layers with HashedLinear.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress(self, model: nn.Module | None = None) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Return (compressed_model, statistics).

`HashedNetsCompressor(model: nn.Module | None = None, n_buckets: int | None = None, compression_ratio: float | None = None, rank: int | float | None = None, max_error: float = 0.35, target_attention_only: bool = False, fallback_to_original: bool = True, max_retries: int = 2, use_sign: bool = True, lora_rank: int = 0, lora_alpha: float = 1.0, freeze_hash_table: bool = False, skip_attention_qkv: bool = False, cfg: Any | None = None, target_layers: list[str] | None = None, **kwargs: Any)`
:   Replace target linear/Conv1D layers with HashedLinear.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.