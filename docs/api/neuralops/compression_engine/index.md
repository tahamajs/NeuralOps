Module neuralops.compression_engine
===================================
Auto-import all compression_engine modules and export their registered classes.

Sub-modules
-----------
* neuralops.compression_engine.a3_compressor
* neuralops.compression_engine.aa_svd_compressor
* neuralops.compression_engine.aatc_xkv_compressor
* neuralops.compression_engine.adaptive_ranks
* neuralops.compression_engine.adasvd_compressor
* neuralops.compression_engine.admm_compressor
* neuralops.compression_engine.admm_pruner
* neuralops.compression_engine.armor_pruner
* neuralops.compression_engine.ast_pruner
* neuralops.compression_engine.astro_quantizer
* neuralops.compression_engine.bonsai_pruner
* neuralops.compression_engine.cachenotes_compressor
* neuralops.compression_engine.come_compressor
* neuralops.compression_engine.compot_compressor
* neuralops.compression_engine.coopq_quantizer
* neuralops.compression_engine.cr_compressor
* neuralops.compression_engine.cur_compressor
* neuralops.compression_engine.cur_moe_compressor
* neuralops.compression_engine.delta_kv
* neuralops.compression_engine.direct_svd
* neuralops.compression_engine.dobi_svd
* neuralops.compression_engine.drone_compressor
* neuralops.compression_engine.dynamic_rank
* neuralops.compression_engine.enlighten_compressor
* neuralops.compression_engine.entropy_budget_allocation
* neuralops.compression_engine.erc_svd
* neuralops.compression_engine.faq_quantizer
* neuralops.compression_engine.flap
* neuralops.compression_engine.flat_llm
* neuralops.compression_engine.frandi_compressor
* neuralops.compression_engine.fwsvd_compressor
* neuralops.compression_engine.gfws_d_compressor
* neuralops.compression_engine.grecodec_compressor
* neuralops.compression_engine.hashed_net_compressor
* neuralops.compression_engine.huffman_coder
* neuralops.compression_engine.hybrid_svd_layer
* neuralops.compression_engine.hybridkv_compressor
* neuralops.compression_engine.kmeans_quantize
* neuralops.compression_engine.kronecker_compressor
* neuralops.compression_engine.kronecker_tt_lora
* neuralops.compression_engine.kvtc_compressor
* neuralops.compression_engine.liftr
* neuralops.compression_engine.lillama_compressor
* neuralops.compression_engine.lsc_compressor
* neuralops.compression_engine.mamba_shedder
* neuralops.compression_engine.mgie_svd
* neuralops.compression_engine.mgie_svd_compressor
* neuralops.compression_engine.mlorq_compressor
* neuralops.compression_engine.modegpt_compressor
* neuralops.compression_engine.nli_softmax
* neuralops.compression_engine.nmf_compressor
* neuralops.compression_engine.oats_compressor
* neuralops.compression_engine.optimal_subspace_projection
* neuralops.compression_engine.proxsparse
* neuralops.compression_engine.replaceme_compressor
* neuralops.compression_engine.res_svd
* neuralops.compression_engine.roast_hashed_linear
* neuralops.compression_engine.rotational_slicegpt_int4
* neuralops.compression_engine.rpca_compressor
* neuralops.compression_engine.saes_svd_v2
* neuralops.compression_engine.salr
* neuralops.compression_engine.sdrm_compressor
* neuralops.compression_engine.slicegpt
* neuralops.compression_engine.slicegpt_pca
* neuralops.compression_engine.slim_compressor
* neuralops.compression_engine.soka_compressor
* neuralops.compression_engine.sparsegpt
* neuralops.compression_engine.subspace_projection
* neuralops.compression_engine.svd_linear
* neuralops.compression_engine.svd_llm_compressor
* neuralops.compression_engine.swift_svd
* neuralops.compression_engine.swift_svd_act
* neuralops.compression_engine.tomoe_compressor
* neuralops.compression_engine.tt_compressor
* neuralops.compression_engine.uniql_compressor
* neuralops.compression_engine.zerokv_neo_compressor

Classes
-------

`A3Compressor(model: nn.Module, rank: int = 64, use_fisher: bool = True, use_energy_weighting: bool | None = None, use_cholesky: bool = False, allow_ln_failure: bool = False, lora_rank: int = 0, lora_alpha: float = 1.0, recalibrate_ln: bool = True, target_attention_only: bool = True, max_error: float = 0.5, max_retries: int = 5, fallback_to_original: bool = True, normalize_scale: bool = True, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, n_calib: int = 500, device: str | torch.device | None = None, cov_dtype: str = 'float32', **kwargs: Any)`
:   A3: Activation-aware factorized SVD compressor.
    
    Uses E[x²] weighting (second moment), optional full Cholesky whitening,
    STRS rank search, and LayerNorm recalibration.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress_linear(self, module: nn.Module, name: str = '', cov: torch.Tensor | None = None, fisher: torch.Tensor | None = None, rank: int | None = None, activation_energy: torch.Tensor | None = None, lora_rank: int = 0, lora_alpha: float = 1.0, X_for_error: torch.Tensor | None = None, **kwargs: Any)`
    :

`A3FactorizedLinear(in_features: int, out_features: int, rank: int, bias: bool = True, dtype: torch.dtype = torch.float32, device: torch.device | None = None, lora_rank: int = 0, lora_alpha: float = 1.0, freeze_base: bool = False)`
:   Low-rank factorized linear with optional LoRA:
        y = (x @ V.T) @ U.T + LoRA(x) + b
    Parameter count: r·(d_in + d_out) instead of d_in·d_out.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, rank: int, U_factor: torch.Tensor | None = None, V_factor: torch.Tensor | None = None, lora_rank: int = 0, lora_alpha: float = 1.0, freeze_base: bool = False) ‑> neuralops.compression_engine.a3_compressor.A3FactorizedLinear`
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

    `freeze_base_weights(self) ‑> None`
    :

    `get_weight(self) ‑> torch.Tensor`
    :

    `reset_parameters(self) ‑> None`
    :

    `set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) ‑> None`
    :

`AASVDCompressor(model: nn.Module, tokenizer: Any | None = None, rank: int = 288, ema_beta: float = 0.9, max_error: float = 0.5, fallback_to_original: bool = True, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, target_modules: list[str] | None = None, target_attention_only: bool = True, force_rank: bool = False, lora_rank: int = 0, lora_alpha: float = 1.0, cov_dtype: str = 'float32', **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress_linear(self, layer: nn.Module, rank: int | None = None, cov: torch.Tensor | None = None, cov_factors: tuple[torch.Tensor, torch.Tensor] | None = None) ‑> tuple[neuralops.compression_engine.aa_svd_compressor.SVDLinear | None, float, int]`
    :

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

`ADMMCompressor(model: nn.Module, sparsity: float = 0.5, quant_bits: int = 4, rho1: float = 0.1, rho2: float = 0.1, max_iter: int = 8, structured_24: bool = True, block_size: tuple[int, int] = (4, 4), max_error: float = 0.45, fallback_to_original: bool = True, target_attention_only: bool = False, target_modules: list[str] | None = None, quant_per_channel: bool = True, comp_config: Any = None, **kwargs: Any)`
:   ADMM-based joint structured pruning + uniform quantization engine.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

`ADMMMaskedLinear(in_features: int, out_features: int, bias: bool = True, dtype: torch.dtype = torch.float32, device: torch.device | None = None)`
:   Linear layer with continuous weight + binary mask and STE gradient.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, mask: torch.Tensor) ‑> neuralops.compression_engine.admm_pruner.ADMMMaskedLinear`
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

`ADMMPruner(model: nn.Module, sparsity: float = 0.5, rho: float = 0.001, max_iter: int = 6, structured_24: bool = True, block_size: tuple[int, int] = (4, 4), dynamic_rho: bool = True, use_ste: bool = True, recalibrate_ln: bool = True, max_error: float = 0.45, fallback_to_original: bool = True, protect_critical: bool = True, target_attention_only: bool = False, target_modules: list[str] | None = None, lambda_distill: float = 0.1, distill_batch_size: int = 32, distill_use_activation: bool = False, distill_lr: float = 0.1, distill_steps: int = 2, calibration_inputs: dict[str, torch.Tensor] | None = None, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Production-hardened ADMM Pruner with Layer-Wise Reconstruction Distillation,
    2:4 semi-structured sparsity, STE gradients, and dynamic rho scheduling.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

`ADMMSTEPrune(*args, **kwargs)`
:   Straight-Through Estimator for ADMM sparse weights.
    Forward: y = weight * mask (with dtype-aligned mask).
    Backward: gradient flows through the identity to weight; mask has no grad.

    ### Ancestors (in MRO)

    * torch.autograd.function.Function
    * torch.autograd.function._SingleLevelFunction
    * torch._C._FunctionBase
    * torch.autograd.function.FunctionCtx
    * torch.autograd.function._HookMixin

    ### Static methods

    `backward(ctx, grad_output: torch.Tensor)`
    :   Define a formula for differentiating the operation with backward mode automatic differentiation.
        
        This function is to be overridden by all subclasses.
        (Defining this function is equivalent to defining the ``vjp`` function.)
        
        It must accept a context :attr:`ctx` as the first argument, followed by
        as many outputs as the :func:`forward` returned (None will be passed in
        for non tensor outputs of the forward function),
        and it should return as many tensors, as there were inputs to
        :func:`forward`. Each argument is the gradient w.r.t the given output,
        and each returned value should be the gradient w.r.t. the
        corresponding input. If an input is not a Tensor or is a Tensor not
        requiring grads, you can just pass None as a gradient for that input.
        
        The context can be used to retrieve tensors saved during the forward
        pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
        of booleans representing whether each input needs gradient. E.g.,
        :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
        first input to :func:`forward` needs gradient computed w.r.t. the
        output.

    `forward(ctx, weight: torch.Tensor, mask: torch.Tensor) ‑> torch.Tensor`
    :   Define the forward of the custom autograd Function.
        
        This function is to be overridden by all subclasses.
        There are two ways to define forward:
        
        Usage 1 (Combined forward and ctx)::
        
            @staticmethod
            def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
                pass
        
        - It must accept a context ctx as the first argument, followed by any
          number of arguments (tensors or other types).
        - See :ref:`combining-forward-context` for more details
        
        Usage 2 (Separate forward and ctx)::
        
            @staticmethod
            def forward(*args: Any, **kwargs: Any) -> Any:
                pass
        
        
            @staticmethod
            def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
                pass
        
        - The forward no longer accepts a ctx argument.
        - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
          staticmethod to handle setting up the ``ctx`` object.
          ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
          to the forward.
        - See :ref:`extending-autograd` for more details
        
        The context can be used to store arbitrary data that can be then
        retrieved during the backward pass. Tensors should not be stored
        directly on `ctx` (though this is not currently enforced for
        backward compatibility). Instead, tensors should be saved either with
        :func:`ctx.save_for_backward` if they are intended to be used in
        ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
        if they are intended to be used for in ``jvp``.

`ARMORLinear(in_features: int, out_features: int, rank: int, bias: bool = True, lora_alpha: float | None = None, dtype: torch.dtype = torch.float32, device: torch.device | None = None)`
:   Decoupled sparse + low-rank residual layer.
    
    Forward:
        y = x @ (sparse_weight ⊙ sparse_mask)^T
            + (scaling) * (x @ lora_A^T) @ lora_B^T
            + bias
    
    Invariants:
      - `sparse_weight` holds the ORIGINAL weight values (not pre-masked).
      - `sparse_mask` (non-persistent) holds a 0/1 mask applied once per forward.
      - `lora_alpha == rank` so `scaling == 1.0`, so the SVD residual is exact.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_linear(linear: nn.Linear | Any, mask: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor, lora_alpha: float | None = None) ‑> neuralops.compression_engine.armor_pruner.ARMORLinear`
    :

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

`ARMORPruner(model: nn.Module, block_size: int = 4, rank: int = 16, sparsity: float = 0.5, structured_24: bool = True, decouple: bool = True, recalibrate_ln: bool = True, lora_alpha: float | None = None, max_error: float = 0.35, fallback_to_original: bool = True, target_attention_only: bool = False, skip_attention_qkv: bool = False, skip_attention_output: bool = False, target_modules: list[str] | None = None, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Production-hardened ARMOR Pruner.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`ASTPrunedLoRALinear(in_features: int, out_features: int, weight: torch.Tensor, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Linear layer with frozen AST-pruned base weights and optional trainable
    LoRA adapters.
    
    The base weight is stored in original coordinate basis with a true 2:4
    (or N:M) sparsity pattern. Framework may pack it as sparse if desired.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, weight: torch.Tensor, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.ast_pruner.ASTPrunedLoRALinear`
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

    `merge(self) ‑> None`
    :   Merge LoRA into base weight in place.

    `merge_and_unload(self) ‑> None`
    :

    `unmerge(self) ‑> None`
    :

`ASTPruner(model: nn.Module, block_size: int = 4, n_keep: int = 2, dim: int = 1, use_pca: bool = True, use_fisher: bool = True, recalibrate_ln: bool = True, max_error: float = 0.2, fallback_to_original: bool = True, protect_critical: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = True, target_attention_only: bool = False, target_modules: list[str] | None = None, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   AST Pruner: vectorized semi-structured 2:4 / N:M pruning with empirical
    Jacobian calibration and PCA coordinate rotation (saliency only).
    
    Design notes:
      * PCA rotation is used **for saliency estimation only**. The actual
        2:4 mask is applied to the ORIGINAL weights so the sparsity pattern
        is preserved and remains exploitable by sparse backends.
      * The Jacobian proxy (0.5·||logits||²) is a heuristic inspired by
        SparseGPT; it is NOT the true Fisher Information Matrix.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

`AdaSVDCompressor(model: nn.Module, rank: int | None = None, target_param_ratio: float = 0.7, num_iters: int = 10, max_error: float = 0.5, fallback_to_original: bool = True, target_attention_only: bool = True, min_rank: int = 16, target_modules: list[str] | None = None, tokenizer: Any | None = None, calibration_texts: list[str] | None = None, use_weighted_als: bool = False, use_mean_shift: bool = True, use_gating_d: bool = True, device: str | torch.device | None = None, comp_config: Any = None, max_calib_tokens: int = 4096, fused_qkv_error_threshold: float = 0.3, gelu_layer_patterns: tuple[str, ...] | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress_linear(self, layer: nn.Module, rank: int, name: str = '', covariance: torch.Tensor | None = None, X_calib: torch.Tensor | None = None, out_variance: torch.Tensor | None = None, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module | None, float, int]`
    :

`AdaptiveRanksCompressor(model: nn.Module, energy_threshold: float = 0.92, min_rank: int = 16, max_error: float = 0.2, fallback_to_original: bool = True, protect_critical: bool = True, target_attention_only: bool = False, target_modules: list[str] | None = None, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, use_jg_svd: bool = False, use_cholesky_jg: bool = True, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   Adaptive Ranks SVD Compressor.
        
        Dynamically determines ranks per layer according to cumulative singular value energy threshold:
            $$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^d \sigma_j^2} \ge \tau \quad (\tau = 0.92)$$
        with minimum rank floor $r_{\min} = \max(16, \lfloor 0.15 \cdot \min(d_{in}, d_{out}) \rfloor, \lfloor 0.10 \cdot d_{out} \rfloor)$,
        activation covariance whitening, and symmetrically distributed singular values:
            $$U_r = U_{:, :r} \sqrt{\Sigma_r}, \quad V_r = (\sqrt{\Sigma_r} V^\top_{:r, :}) \cdot \operatorname{Cov}(X)^{-1/2}$$
        
        Production hardening features:
        - Relative Frobenius reconstruction error validation with configurable max_error (default 0.20).
        - Safe fallback to original weights by default (fallback_to_original=True).
        - Robust Conv1D and LoRALinear support via base_layer weight transposition.
        - SVD and covariance eigendecomposition computed strictly in FP32 with NaN/Inf guards.
        - Unified critical-layer protection via should_compress_linear.
        - Adaptive rank retry when reconstruction error exceeds threshold.
        - Parameter reduction verification: ensures r * (in + out) < in * out.
        - Comprehensive per-layer statistics and logging.

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress target layers with activation covariance whitening and safe validation.

    `compress_linear(self, layer: nn.Module, cov: torch.Tensor | None = None, layer_name: str | None = None) ‑> tuple[neuralops.compression_engine.aa_svd_compressor.SVDLinear | None, float, int, bool, bool]`
    :   Decompose a single linear, Conv1D, or LoRALinear layer via energy-thresholded SVD
        with covariance whitening, returning (svd_layer, relative_error, rank, used_cov).

`AdaptiveTransformerCompressor(model: nn.Module, energy_threshold: float = 0.92, min_rank: int = 16, max_error: float = 0.2, fallback_to_original: bool = True, protect_critical: bool = True, target_attention_only: bool = False, target_modules: list[str] | None = None, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, use_jg_svd: bool = False, use_cholesky_jg: bool = True, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   Adaptive Ranks SVD Compressor.
        
        Dynamically determines ranks per layer according to cumulative singular value energy threshold:
            $$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^d \sigma_j^2} \ge \tau \quad (\tau = 0.92)$$
        with minimum rank floor $r_{\min} = \max(16, \lfloor 0.15 \cdot \min(d_{in}, d_{out}) \rfloor, \lfloor 0.10 \cdot d_{out} \rfloor)$,
        activation covariance whitening, and symmetrically distributed singular values:
            $$U_r = U_{:, :r} \sqrt{\Sigma_r}, \quad V_r = (\sqrt{\Sigma_r} V^\top_{:r, :}) \cdot \operatorname{Cov}(X)^{-1/2}$$
        
        Production hardening features:
        - Relative Frobenius reconstruction error validation with configurable max_error (default 0.20).
        - Safe fallback to original weights by default (fallback_to_original=True).
        - Robust Conv1D and LoRALinear support via base_layer weight transposition.
        - SVD and covariance eigendecomposition computed strictly in FP32 with NaN/Inf guards.
        - Unified critical-layer protection via should_compress_linear.
        - Adaptive rank retry when reconstruction error exceeds threshold.
        - Parameter reduction verification: ensures r * (in + out) < in * out.
        - Comprehensive per-layer statistics and logging.

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress target layers with activation covariance whitening and safe validation.

    `compress_linear(self, layer: nn.Module, cov: torch.Tensor | None = None, layer_name: str | None = None) ‑> tuple[neuralops.compression_engine.aa_svd_compressor.SVDLinear | None, float, int, bool, bool]`
    :   Decompose a single linear, Conv1D, or LoRALinear layer via energy-thresholded SVD
        with covariance whitening, returning (svd_layer, relative_error, rank, used_cov).

`Any(*args, **kwargs)`
:   Special type indicating an unconstrained type.
    
    - Any is compatible with every type.
    - Any assumed to have all methods.
    - All values assumed to be instances of Any.
    
    Note that all the above statements are true from the point of view of
    static type checkers. At runtime, Any should not be used with instance
    checks.

`AstroLinear(in_features: int, out_features: int, bias: bool = True, bits: int = 8, symmetric: bool = True, per_channel: bool = False, per_channel_bits: torch.Tensor | None = None)`
:   Linear layer using Astro uniform quantization with STE backpropagation support.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, bits: int = 8, symmetric: bool = True, per_channel: bool = False, per_channel_bits: torch.Tensor | None = None) ‑> neuralops.compression_engine.astro_quantizer.AstroLinear`
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

`AstroQuant4Bit(bits: int = 4)`
:   Non-uniform quantizer with INT2 / INT4 / INT8 codebooks optimized for
    Gaussian weight distributions.
    
    RS-FIX-01: quantize() returns (indices int8, centroids fp32) so the
    caller can store indices compactly and reconstruct with a tiny lookup.

    ### Class variables

    `SUPPORTED_BITS`
    :   The type of the None singleton.

    ### Static methods

    `dequantize(indices: torch.Tensor, centroids: torch.Tensor) ‑> torch.Tensor`
    :   Reconstruct floats from indices and centroids.

    ### Methods

    `quantize(self, w: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
    :   Symmetric per-tensor quantization.
        
        Returns:
            indices:   (..., ) int8 tensor of centroid indices in [0, C)
            centroids: (C,) fp32 tensor of scaled centroids

`AstroQuantizer(model: nn.Module, bits: int = 8, symmetric: bool = True, per_channel: bool = False, use_ste: bool = False, max_error: float = 0.2, fallback_to_original: bool = True, protect_critical: bool = True, target_modules: list[str] | None = None, target_attention_only: bool = False, use_fisher_adaptive_bits: bool = False, target_avg_bits: float = 6.0, min_bits: int = 2, max_bits: int = 8, fisher_beta: float = 0.5, **kwargs: Any)`
:   AstroQuantizer: Uniform min-max / symmetric quantization engine with Fisher-adaptive bit allocation.
    
    Production hardening features:
    - Support for LoRALinear wrappers, nn.Linear, and Conv1D (GPT-2).
    - Novel Fisher-Adaptive Bit Allocation allocating mixed-precision bit-widths [min_bits, max_bits] per output channel.
    - target_attention_only parameter skipping FFN/MLP and preserving classification heads.
    - Robust config handling (_cfg_get) supporting None, dict, and namespace objects.
    - Fallback freezing: parameters in any fallback layer are frozen (requires_grad=False).
    - Full FP32 quantization calculation with NaN/Inf guards.
    - Straight-Through Estimator (STE) autograd flow.
    - Transparent metric tracking: actual_average_bits, effective_compression_rate, and theoretical_vram_saving.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Quantize all target linear/Conv1D layers in the model and return (compressed_model, stats).

`BaseAttention(config: Any | None = None, rank: int | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Descendants

    * neuralops.attentions.adaptive_span_attention.AdaptiveSpanAttention
    * neuralops.attentions.adaptive_sparse_attention.AdaptiveSparseAttention
    * neuralops.attentions.adaptive_svd_attention.AdaptiveSVDAttention
    * neuralops.attentions.adasplash_attention.AdaSplashAttention
    * neuralops.attentions.aegis_hybrid.AegisHybridAttention
    * neuralops.attentions.aegis_spectron_fp8.AegisSpectronFP8Attention
    * neuralops.attentions.aegis_v2.AegisAttentionV2
    * neuralops.attentions.aegis_v5.AegisAttention
    * neuralops.attentions.aegis_v6.AegisAttentionV3
    * neuralops.attentions.asvd_attention.ASVDAttention
    * neuralops.attentions.bam_umoe.BAMUMoEBlock
    * neuralops.attentions.block_transformer_hiba.BlockTransformerHIBA
    * neuralops.attentions.chela_slay.ChelaSlayAttention
    * neuralops.attentions.cluster_attention.ClusterAttention
    * neuralops.attentions.compressed_attention.CompressedAttentionLayer
    * neuralops.attentions.compressed_mha_attention.CompressedDistilBertAttention
    * neuralops.attentions.cosformer.CosformerAttention
    * neuralops.attentions.decay_linear_attention.DecayLinearAttention
    * neuralops.attentions.eat.EATBlock
    * neuralops.attentions.expert_choice_moe.ExpertChoiceMoEAttention
    * neuralops.attentions.factorised_matrix_state.FactorisedMatrixState
    * neuralops.attentions.flash_attention3.FlashAttention3
    * neuralops.attentions.flash_attention3_fp8.FlashAttention3FP8
    * neuralops.attentions.full_attention.FullAttention
    * neuralops.attentions.fused_svd_kernel.FusedSVDKernelAttention
    * neuralops.attentions.fusion_attn.FUSIONATTN
    * neuralops.attentions.galerkin_attention.GalerkinAttention
    * neuralops.attentions.gated_delta_net.GatedDeltaNet
    * neuralops.attentions.gated_deltanet_block.GatedDeltaNetBlock
    * neuralops.attentions.gated_linear_attention.GatedLinearAttention
    * neuralops.attentions.goldfinch_ahn.GoldFinchAHNBlock
    * neuralops.attentions.gqa_attention.GroupedQueryAttention
    * neuralops.attentions.granite_hybrid.GraniteHybridBlock
    * neuralops.attentions.griffin_ssd.GriffinSSDBlock
    * neuralops.attentions.halo_regla.HALOReGLAAttention
    * neuralops.attentions.hamming_attention.HammingAttention
    * neuralops.attentions.hgrn2_gated_deltanet.HGRN2GatedDeltaNet
    * neuralops.attentions.hgrn2_layer.HGRN2Layer
    * neuralops.attentions.hgrn2_outer.HGRN2OuterBlock
    * neuralops.attentions.hla_attention.HLAAttention
    * neuralops.attentions.hybrid_attention.HybridAttention
    * neuralops.attentions.hymba_transx.HymbaTransXBlock
    * neuralops.attentions.infini_attention.InfiniAttention
    * neuralops.attentions.jamba.Jamba2Block
    * neuralops.attentions.jamba_info.JambaInfoBlock
    * neuralops.attentions.kda_attention.KDAAttention
    * neuralops.attentions.leka_v1.LEKAV1Attention
    * neuralops.attentions.leka_v2.LEKAv2Attention
    * neuralops.attentions.leka_v3.LEKAV3Attention
    * neuralops.attentions.linformer_attention.LinformerAttention
    * neuralops.attentions.lola_attention.LoLAAttention
    * neuralops.attentions.lora_attention.LoRAAttention
    * neuralops.attentions.lsh_attention.LSHAttention
    * neuralops.attentions.mamba_svd_flash.MambaSVDFlash
    * neuralops.attentions.minimal_nova.MinimalNOVAAttention
    * neuralops.attentions.minimal_svd.MinimalSVDAttention
    * neuralops.attentions.mixattention_moska.MixAttentionMoSKA
    * neuralops.attentions.mla_attention.MLAAttention
    * neuralops.attentions.moba_attention.MoBAAttention
    * neuralops.attentions.nasc_attention.NASCAttention
    * neuralops.attentions.nexus.NEXUSAttention
    * neuralops.attentions.nha_attention.NativeHybridAttention
    * neuralops.attentions.nsa_attention.NativeSparseAttention
    * neuralops.attentions.nystrom_attention.NystromAttention
    * neuralops.attentions.optimized_hybrid_attention.OptimizedHybridAttention
    * neuralops.attentions.orion.ORIONAttention
    * neuralops.attentions.padre_attention.PADReAttention
    * neuralops.attentions.performer_attention.PerformerAttention
    * neuralops.attentions.phantom.PHANTOMAttention
    * neuralops.attentions.primal_attention.PrimalAttention
    * neuralops.attentions.progressive_token_pruning.ProgressiveTokenPruning
    * neuralops.attentions.qcqa_attention.QCQAAttention
    * neuralops.attentions.rala_attention.RALALinearAttention
    * neuralops.attentions.reap_moe.REAPMoEAttention
    * neuralops.attentions.regla_attention.ReGLAAttention
    * neuralops.attentions.replace.ReplaceAttention
    * neuralops.attentions.res_svd.ResSVDAttention
    * neuralops.attentions.routing_attention.RoutingAttention
    * neuralops.attentions.rpc_attention.RPCAttention
    * neuralops.attentions.salt_attention.SALTAttention
    * neuralops.attentions.samba_moe.SambaMoEBlock
    * neuralops.attentions.selective_ssm.SelectiveSSM
    * neuralops.attentions.selective_ssm.SelectiveSSM
    * neuralops.attentions.soa_attn.SOAAttention
    * neuralops.attentions.span_expanded_attention.SpanExpandedAttention
    * neuralops.attentions.spectra_v2.SpectraV2Attention
    * neuralops.attentions.spectraformer_attention.SpectraformerAttention
    * neuralops.attentions.spectral_mamba.SpectralMambaAttention
    * neuralops.attentions.spectron.SpectronAttention
    * neuralops.attentions.squeezing_heads_distill.SqueezingHeadsDistiller
    * neuralops.attentions.svd_attention.SVDLinearAttention
    * neuralops.attentions.svd_compressed.SVDCompressedAttention
    * neuralops.attentions.svd_linear_attention.SVDLinearAttention
    * neuralops.attentions.svd_llm_v2_attention.SVDLLMV2Attention
    * neuralops.attentions.svd_mamba.SVDMamba
    * neuralops.attentions.svd_projection_attention.SVDProjectionAttention
    * neuralops.attentions.template_attention.YourCustomAttention
    * neuralops.attentions.tlinformer_attention.TLinFormerAttention
    * neuralops.attentions.token_compressed_attention.TokenCompressedAttention
    * neuralops.attentions.tpa_attention.TPAttention
    * neuralops.attentions.triton_cosformer.TritonCosformerAttention
    * neuralops.attentions.triton_svd_linear_attention.TritonSVDLinearAttention
    * neuralops.attentions.unified_rope.UnifiedRoPE
    * neuralops.attentions.upi_stuffed_mamba.SelectiveMamba
    * neuralops.attentions.vig_attention.ViGAttention
    * neuralops.attentions.vspec.VSpecAttention
    * neuralops.attentions.wgqa_attention.WeightedGroupedQueryAttention
    * neuralops.attentions.xattention.XAttention
    * neuralops.attentions.zeros_attention.ZeroSAttention
    * neuralops.compression_engine.nli_softmax.NLISoftmaxAttention
    * neuralops.core.replace_attention.ReplaceAttention

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `extra_repr(self) ‑> str`
    :   Extra repr.

    `forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs) ‑> tuple[typing.Any, Any | None]`
    :   Must return (output, attention_weights_or_None).

`BaseCompressor(model: torch.nn.modules.module.Module, name: str = 'base', **kwargs)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Descendants

    * neuralops.compression_engine.a3_compressor.A3Compressor
    * neuralops.compression_engine.aa_svd_compressor.AASVDCompressor
    * neuralops.compression_engine.aatc_xkv_compressor.AATC_XKVCompressor
    * neuralops.compression_engine.adaptive_ranks.AdaptiveRanksCompressor
    * neuralops.compression_engine.adasvd_compressor.AdaSVDCompressor
    * neuralops.compression_engine.admm_compressor.ADMMCompressor
    * neuralops.compression_engine.admm_pruner.ADMMPruner
    * neuralops.compression_engine.armor_pruner.ARMORPruner
    * neuralops.compression_engine.ast_pruner.ASTPruner
    * neuralops.compression_engine.astro_quantizer.AstroQuantizer
    * neuralops.compression_engine.bonsai_pruner.BonsaiPruner
    * neuralops.compression_engine.cachenotes_compressor.CacheNotesCompressor
    * neuralops.compression_engine.come_compressor.CoMeCompressor
    * neuralops.compression_engine.compot_compressor.COMPOTCompressor
    * neuralops.compression_engine.coopq_quantizer.CoopQQuantizer
    * neuralops.compression_engine.cr_compressor.CRCompressor
    * neuralops.compression_engine.cur_compressor.CURCompressor
    * neuralops.compression_engine.cur_moe_compressor.CURMoECompressor
    * neuralops.compression_engine.delta_kv.DeltaKVCompressor
    * neuralops.compression_engine.direct_svd.DirectSVDCompressor
    * neuralops.compression_engine.dobi_svd.DobiSVDCompressor
    * neuralops.compression_engine.drone_compressor.DRONECompressor
    * neuralops.compression_engine.dynamic_rank.DynamicEntropySVDCompressor
    * neuralops.compression_engine.enlighten_compressor.ENLightenCompressor
    * neuralops.compression_engine.entropy_budget_allocation.EntropyBudgetCompressor
    * neuralops.compression_engine.erc_svd.ERCCompressor
    * neuralops.compression_engine.faq_quantizer.FAQQuantizer
    * neuralops.compression_engine.flap.FLAPCompressor
    * neuralops.compression_engine.flat_llm.FLATLLMCompressor
    * neuralops.compression_engine.frandi_compressor.FRanDICompressor
    * neuralops.compression_engine.fwsvd_compressor.FWSVDCompressor
    * neuralops.compression_engine.gfws_d_compressor.GFWSCompressor
    * neuralops.compression_engine.grecodec_compressor.GreCoDeCCompressor
    * neuralops.compression_engine.hashed_net_compressor.HashedNetCompressor
    * neuralops.compression_engine.huffman_coder.HuffmanCompressor
    * neuralops.compression_engine.hybrid_svd_layer.HybridSVDCompressor
    * neuralops.compression_engine.hybridkv_compressor.HybridKVCompressor
    * neuralops.compression_engine.kmeans_quantize.KMeansCompressor
    * neuralops.compression_engine.kronecker_compressor.KroneckerCompressor
    * neuralops.compression_engine.kronecker_tt_lora.KroneckerTTLoRACompressor
    * neuralops.compression_engine.kvtc_compressor.KVTCCompressor
    * neuralops.compression_engine.liftr.LIFTRCompressor
    * neuralops.compression_engine.lillama_compressor.LiLLaMADistiller
    * neuralops.compression_engine.lsc_compressor.LSCCompressor
    * neuralops.compression_engine.mamba_shedder.MambaShedderCompressor
    * neuralops.compression_engine.mgie_svd.MGIECompressor
    * neuralops.compression_engine.mgie_svd_compressor.MGIESVDBudgetCompressor
    * neuralops.compression_engine.mlorq_compressor.MLoRQCompressor
    * neuralops.compression_engine.modegpt_compressor.MoDeGPTCompressor
    * neuralops.compression_engine.nli_softmax.NLISoftmaxCompressor
    * neuralops.compression_engine.nmf_compressor.NMFCompressor
    * neuralops.compression_engine.oats_compressor.OATSCompressor
    * neuralops.compression_engine.optimal_subspace_projection.OptimalSubspaceProjectionCompressor
    * neuralops.compression_engine.proxsparse.ProxSparsePruner
    * neuralops.compression_engine.replaceme_compressor.ReplaceMeCompressor
    * neuralops.compression_engine.res_svd.ResSVDCompressor
    * neuralops.compression_engine.roast_hashed_linear.ROASTCompressor
    * neuralops.compression_engine.rotational_slicegpt_int4.RotationalSliceGPTINT4Compressor
    * neuralops.compression_engine.rpca_compressor.RPCACompressor
    * neuralops.compression_engine.saes_svd_v2.SAESSVDCompressorV2
    * neuralops.compression_engine.salr.SALRCompressor
    * neuralops.compression_engine.sdrm_compressor.SDRMCompressor
    * neuralops.compression_engine.slicegpt.SliceGPTCompressor
    * neuralops.compression_engine.slicegpt_pca.SliceGPTPCACompressor
    * neuralops.compression_engine.slim_compressor.SLiMCompressor
    * neuralops.compression_engine.soka_compressor.LowRankCompressor
    * neuralops.compression_engine.sparsegpt.SparseGPTCompressor
    * neuralops.compression_engine.subspace_projection.SubspaceProjectionCompressor
    * neuralops.compression_engine.svd_linear.SVDLinearCompressor
    * neuralops.compression_engine.svd_llm_compressor.SVDLLMCompressor
    * neuralops.compression_engine.swift_svd.SwiftSVDCompressor
    * neuralops.compression_engine.swift_svd_act.SwiftSVDActivationCompressor
    * neuralops.compression_engine.tomoe_compressor.ToMoECompressor
    * neuralops.compression_engine.tt_compressor.TTCompressor
    * neuralops.compression_engine.uniql_compressor.UniQLCompressor
    * neuralops.compression_engine.zerokv_neo_compressor.ZeroKVNeoCompressor

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Return (compressed_model, statistics).

    `get_compression_stats(self) ‑> dict[str, typing.Any]`
    :   Return the compression stats.

    `summary(self) ‑> str`
    :   Summary.

`BonsaiPrunedLoRALinear(in_features: int, out_features: int, bias: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, freeze_base: bool = True, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Linear with sparse pruned base weight (frozen) + optional trainable LoRA adapter.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_linear(linear: nn.Linear | Any, pruned_weight: torch.Tensor, mask: torch.Tensor, lora_r: int = 0, lora_alpha: float = 1.0, freeze_base: bool = True) ‑> neuralops.compression_engine.bonsai_pruner.BonsaiPrunedLoRALinear`
    :

    ### Instance variables

    `effective_weight: torch.Tensor`
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

`BonsaiPruner(model: nn.Module, sparsity: float = 0.5, steps: int = 8, recovery_steps: int = 3, recovery_lr: float = 0.001, structured_24: bool = False, block_size: tuple[int, int] = (4, 4), protect_cls: bool = True, skip_attention_qkv: bool = True, target_attention_only: bool = False, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, adaptive_sparsity: bool = False, min_numel: int = 16, max_error: float | None = None, fallback_to_original: bool = True, recalibrate_ln: bool = True, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, cfg: Any | None = None, max_recovery_steps: int = 32, max_step_size: float = 0.01, **kwargs: Any)`
:   Bonsai Pruner: progressive multi-step iterative pruning with cubic schedule,
    inter-step gradient descent recovery, and [CLS]/GQA protection.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

`COMPOTCompressor(model: nn.Module | None = None, rank: int = 64, max_error: float = 0.2, fallback_to_original: bool = True, calibration_texts: list[str] | None = None, calibration_loader: Any | None = None, tokenizer: Any | None = None, cfg: Any | None = None, target_attention_only: bool = False, target_layers: list[str] | None = None, **kwargs: Any)`
:   COMPOT (Compressed Orthogonal Dictionary Learning) Compressor.
    Learns a low-rank basis from activation QR decompositions and compresses
    linear and Conv1D layers into factorized COMPOTLinear modules.
    
    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - Safe fallback to original layer if error exceeds threshold or decomposition is unstable.
    - SVD fallback if QR decomposition on activations fails.
    - Parameter reduction guarantee: checks r * (in + out) < in * out.
    - Robust Conv1D support (GPT-2 c_attn, c_proj) via weight transposition.
    - Unified critical-layer protection via should_compress_linear.
    - Comprehensive per-layer stats and error reporting.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, model: nn.Module | None = None) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress linear and Conv1D layers via COMPOT orthogonal dictionary learning.

`COMPOTLinear(in_features: int, out_features: int, rank: int = 64, bias: bool = True, dtype: torch.dtype | None = None, device: torch.device | None = None)`
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
    :   Factorized Linear Layer constructed via Orthogonal Dictionary Learning (COMPOT):
        W ≈ B @ A
        - A in R^{rank x in_features}: input-orthogonal projection matrix
        - B in R^{out_features x rank}: output dictionary matrix
        
        Forward pass:
        y = ((x @ A^T) @ B^T) + bias

    ### Static methods

    `from_linear_and_activations(layer: nn.Module, activations: torch.Tensor | None = None, rank: int = 64) ‑> tuple[neuralops.compression_engine.compot_compressor.COMPOTLinear, float, bool]`
    :   Construct COMPOTLinear by learning orthogonal dictionary from activations or weights.
        Returns: (compot_layer, relative_reconstruction_error, used_activations)

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix (out_features, in_features).

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Forward pass matching standard Linear and Conv1D.

    `get_weight(self) ‑> torch.Tensor`
    :   Materialize full weight matrix W = B @ A.

`CPTGenerator(hidden_dim: int = 128, num_freqs: int = 8, vocab_size: int = 5000)`
:   Learnable Task-Aware Compression Planning Token (CPT) Generator.
    Maps arbitrary task descriptions to continuous per-feature importance vectors for weight SVD
    and KV cache compression across arbitrary dimension sizes d.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, task: str, d: int, device: torch.device | None = None) ‑> torch.Tensor`
    :   Generate a per-feature importance modulation vector of size d in [0.1, 2.0].

`CRCompressor(model: nn.Module | None = None, rank: int = 64, max_error: float = 0.2, fallback_to_original: bool = True, max_retries: int = 2, target_attention_only: bool = False, cfg: Any | None = None, target_layers: list[str] | None = None, **kwargs: Any)`
:   CR (Column-Row / Interpolative Decomposition) Compressor.
    Decomposes linear layers into skeleton column bases and interpolative projections.
    
    Args:
        model: PyTorch model.
        rank: Rank parameter (number of skeleton columns).
        max_error: Maximum relative Frobenius reconstruction error (default 0.20).
        fallback_to_original: If True, keep original layer when error exceeds max_error (default True).
        max_retries: Number of rank increase retries before fallback (default 2).
        target_attention_only: If True, restrict compression to attention projections only.
        cfg: Optional configuration object.
        target_layers: Optional target layer filter.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, model: nn.Module | None = None) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress linear layers using interpolative decomposition.

    `compress_linear(self, linear: nn.Linear | Any) ‑> neuralops.compression_engine.cr_compressor.CRLinear`
    :   Compress a single linear layer into CRLinear.

`CRLinear(in_features: int, out_features: int, rank: int = 64, bias: bool = True, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Interpolative Decomposition / Column-Row Factorized Linear Layer.
    W ≈ C @ P where:
    - C in R^{out_features x rank}: column skeleton from W
    - P in R^{rank x in_features}: interpolative projection matrix
    
    Forward pass:
    y = ((x @ P^T) @ C^T) + bias
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Linear | Any, rank: int = 64) ‑> neuralops.compression_engine.cr_compressor.CRLinear`
    :   Perform Column Interpolative Decomposition on linear or Conv1D layer weight matrix.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix (out_features, in_features).

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Sub-quadratic forward execution:
        y = ((x @ P^T) @ C^T) + bias

    `get_weight(self) ‑> torch.Tensor`
    :   Materialize full reconstructed weight matrix.

`CURCompressor(model: nn.Module | None = None, rank: int = 64, min_rank: int = 8, method: str = 'leverage', deterministic: bool = True, max_error: float = 0.2, fallback_to_original: bool = True, max_retries: int = 2, lora_r: int = 0, lora_alpha: float = 1.0, freeze_skeletons: bool = False, target_attention_only: bool = False, target_names: list[str] | None = None, exclude_names: list[str] | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Compressor that replaces eligible Linear and Conv1D layers with CURLinear.
    
    Args:
        model: PyTorch model.
        rank: Target rank for skeleton column and row selection (default 64).
        min_rank: Minimum allowable rank for compression (default 8).
        method: Selection method ('leverage' or 'energy', default 'leverage').
        deterministic: Deterministic top-k selection if True (default True).
        max_error: Maximum relative Frobenius reconstruction error allowed (default 0.20).
        fallback_to_original: If True, retains original layer when reconstruction error exceeds max_error (default True).
        max_retries: Number of adaptive rank retries when error threshold is exceeded (default 2).
        lora_r: LoRA adapter rank if adding fine-tuning adapters (default 0).
        lora_alpha: LoRA scaling factor (default 1.0).
        freeze_skeletons: If True, freeze C and R parameters during post-compression training (default False).
        target_attention_only: If True, restrict compression to attention projections only.
        target_names: Optional list of layer name substrings to target.
        exclude_names: Optional list of layer name substrings to exclude.
        cfg: Optional configuration object.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, model: nn.Module | None = None) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Replace eligible Linear and Conv1D layers with CURLinear.

    `compress_linear(self, linear: nn.Linear | Any) ‑> neuralops.compression_engine.cur_compressor.CURLinear`
    :   Compress a single linear layer into CURLinear.

`CURLinear(in_features: int, out_features: int, rank: int = 64, bias: bool = True, method: str = 'leverage', lora_r: int = 0, lora_alpha: float = 1.0, freeze_skeletons: bool = False, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Linear layer whose weight matrix is factorized via CUR cross-approximation:
    W ≈ C @ U @ R
    - C in R^{out_features x rank}: selected columns of W
    - U in R^{rank x rank}: Moore-Penrose pseudo-inverse core matrix
    - R in R^{rank x in_features}: selected rows of W
    
    Executes forward pass in factorized form:
    y = ((x @ R^T) @ U^T) @ C^T + bias
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Linear | Any, rank: int = 64, method: str = 'leverage', deterministic: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, freeze_skeletons: bool = False) ‑> neuralops.compression_engine.cur_compressor.CURLinear`
    :   Construct CURLinear from existing nn.Linear or Conv1D using leverage scores or energy sampling.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix for compatibility.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Sub-quadratic forward execution:
        x: (..., in_features)
        y = ((x @ R^T) @ U^T) @ C^T + bias

    `get_weight(self) ‑> torch.Tensor`
    :   Materialize reconstructed weight matrix W = C @ U @ R.

`CURLowRankLinear(in_features: int, out_features: int, rank: int = 64, bias: bool = True, method: str = 'leverage', lora_r: int = 0, lora_alpha: float = 1.0, freeze_skeletons: bool = False, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Linear layer whose weight matrix is factorized via CUR cross-approximation:
    W ≈ C @ U @ R
    - C in R^{out_features x rank}: selected columns of W
    - U in R^{rank x rank}: Moore-Penrose pseudo-inverse core matrix
    - R in R^{rank x in_features}: selected rows of W
    
    Executes forward pass in factorized form:
    y = ((x @ R^T) @ U^T) @ C^T + bias
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Linear | Any, rank: int = 64, method: str = 'leverage', deterministic: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, freeze_skeletons: bool = False) ‑> neuralops.compression_engine.cur_compressor.CURLinear`
    :   Construct CURLinear from existing nn.Linear or Conv1D using leverage scores or energy sampling.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix for compatibility.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Sub-quadratic forward execution:
        x: (..., in_features)
        y = ((x @ R^T) @ U^T) @ C^T + bias

    `get_weight(self) ‑> torch.Tensor`
    :   Materialize reconstructed weight matrix W = C @ U @ R.

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

`CacheNotesCompressor(model: nn.Module, task_type: str = 'qa', task_description: str | None = None, use_learned_cpt: bool = True, keep_ratio: float = 0.5, max_error: float = 0.35, fallback_to_original: bool = True, protect_critical: bool = True, target_attention_only: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = True, recalibrate_ln: bool = True, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, target_modules: list[str] | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   CacheNotes Compressor: Decouples static weight compression from KV cache dynamic pruning
        with continuous learnable CPT Generator.
        
        Production hardening features:
        - Trainable CPTGenerator mapping task descriptions to per-feature importance vectors.
        - Support for LoRALinear wrappers, nn.Linear, and Conv1D (GPT-2).
        - Preserves existing LoRALinear wrappers without creating redundant layers.
        - target_attention_only parameter skipping FFN/MLP and preserving classification heads.
        - Robust config handling (_cfg_get) supporting None, dict, and namespace objects.
        - Fallback freezing: parameters in any fallback layer are frozen (requires_grad=False).
        - Checkpoint saving and loading for CPTGenerator.
        - Comprehensive metric tracking: actual ranks, reconstruction error, and KV cache saving.

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress linear layers using task-guided CPT-weighted low-rank SVD.

    `get_importance(self, d: int, device: torch.device) ‑> torch.Tensor`
    :   Retrieves feature importance vector using learned CPT generator if trained/active,
        otherwise falling back to static heuristic.

    `load_cpt_generator(self, path: str | Path) ‑> bool`
    :   Load CPT generator state dict from file.

    `save_cpt_generator(self, path: str | Path) ‑> None`
    :   Save CPT generator state dict to file.

    `train_cpt_generator(self, task_descriptions: list[str] | None = None, sample_layers: list[tuple[str, nn.Module]] | None = None, epochs: int = 5, lr: float = 0.0001) ‑> float`
    :   Train CPT generator on sample layers across diverse task descriptions,
        minimizing reconstruction error after CPT-weighted SVD compression.

`CacheNotesKVCache(max_cache_len: int = 128, task_type: str = 'qa', keep_ratio: float = 0.3, num_sinks: int = 4, cpt_generator: CPTGenerator | None = None)`
:   Runtime task-guided dynamic KV cache compressor.
    Evaluates token task-relevance dynamically and prunes non-essential context tokens,
    supporting variable sequence lengths during prefill and autoregressive decoding.

    ### Methods

    `reset(self) ‑> None`
    :   Clear cache state to prevent memory leaks across sequences.

    `update_and_compress(self, new_k: torch.Tensor, new_v: torch.Tensor, is_new_sequence: bool = False) ‑> tuple[torch.Tensor, torch.Tensor]`
    :   Append new KV tokens and apply task-aware retention when exceeding capacity.
        Shapes: (B, H, N, D).
        Automatically resets cache on new multi-token prompt prefill (N > 1) or when is_new_sequence=True (H6).

`CacheNotesLoRALinear(in_features: int, out_features: int, weight: torch.Tensor, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Linear layer containing CacheNotes compressed base weights (frozen)
    with optional trainable LoRA adapters (lora_A, lora_B).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, weight: torch.Tensor, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.cachenotes_compressor.CacheNotesLoRALinear`
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

`CoMeBridge(hidden_size: int, bridge_rank: int = 64, alpha_init: float = 0.1)`
:   Learnable low-rank interpolation adapter bridging representation gaps
    between non-consecutive layers.
    
        W_bridge = I + B @ A     (B: hidden x r, A: r x hidden)
        y        = x + alpha * (W_bridge x - x)
                 = x + alpha * (B @ A) x
    
    At initialization B = 0, so the bridge is exactly the identity and only
    alpha influences the output; gradient flows to A and alpha from step 1.
    Robustly handles Tensor, tuple, list, dict, and BaseModelOutput returns.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, hidden_states: Any, *args: Any, **kwargs: Any) ‑> Any`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`CoMeBridgedLayer(layer: nn.Module, bridge: CoMeBridge | None = None)`
:   Wraps a retained transformer layer with a learnable CoMeBridge.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, hidden_states: Any = None, *args: Any, **kwargs: Any) ‑> Any`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`CoMeCompressor(model: nn.Module, tokenizer: Any | None = None, target_layers: int | None = None, sparsity: float = 0.1, bridge_rank: int | None = None, alpha_init: float = 0.1, lora_r: int = 16, lora_alpha: float = 1.0, skip_attention_qkv: bool = True, protect_critical: bool = True, target_attention_only: bool = False, recalibrate_ln: bool = True, allow_forward_failure: bool = False, device: str | torch.device | None = None, target_modules: list[str] | None = None, rank: int | None = None, **kwargs: Any)`
:   CoMe Compressor: Layer shedding with learnable low-rank interpolation
    adapters (bridges) between non-consecutive retained layers.
    
    Config parameters (via kwargs or comp_config):
        rank                  : int, optional — SVD-style rank. If provided and
                                bridge_rank/sparsity are not explicitly set,
                                they are derived from rank via safe mappings.
        target_layers         : int, optional — target layer count after shedding.
        sparsity              : float, default 0.10 — fraction of layers to remove.
        bridge_rank           : int, optional — bottleneck rank for the
                                low-rank interpolation bridge (B @ A). If not
                                set, derived from rank (rank // 4) or defaults
                                to 64 for the full hidden size.
        alpha_init            : float, default 0.1 — initial bridge mixing coeff.
        lora_r                : int, default 16 — post-shedding LoRA rank.
        lora_alpha            : float, default 1.0 — LoRA scaling.
        target_attention_only : bool, default False.
        skip_attention_qkv    : bool, default True — protect asymmetric GQA.
        recalibrate_ln        : bool, default True.
        allow_forward_failure : bool, default False.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `get_trainable_params(self) ‑> list[torch.nn.parameter.Parameter]`
    :

`CoMeLoRALinear(in_features: int, out_features: int, weight: torch.Tensor, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Linear layer: frozen base weights + optional trainable LoRA adapters.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.come_compressor.CoMeLoRALinear`
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

`CompressedAttentionLayer(config: Any | None = None, rank: int | None = None, rank_qkv: int = 256, rank_out: int = 256, lora_rank: int = 16, original_layer: Any = None, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Descendants

    * neuralops.attentions.compressed_attention.CompressedAttentionLayer2

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Instance variables

    `out_weight: torch.Tensor`
    :   Reconstruct the effective base Out weight from SVD factors.

    `qkv_weight: torch.Tensor`
    :   Reconstruct the effective base QKV weight from SVD factors.

    ### Methods

    `reset_lora_parameters(self) ‑> None`
    :

`CompressionPlanningTokens()`
:   Static CPT heuristic importance generator.
    Serves as safe baseline fallback when learnable CPT generator is inactive.

    ### Class variables

    `TASK_KEYWORD_MAP`
    :   The type of the None singleton.

    ### Static methods

    `generate_importance(task_type: str, num_features: int, device: torch.device = device(type='cpu')) ‑> torch.Tensor`
    :   Generate per-feature importance modulation vector based on static heuristics.

`Conv1D(nf, nx)`
:   1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    
    Basically works like a linear layer but the weights are transposed.
    
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x) ‑> Callable[..., typing.Any]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`CoopQLinear(in_features: int, out_features: int, bias: bool = True, bits: int = 4, symmetric: bool = False, per_channel: bool = True, lora_r: int = 0, lora_alpha: float = 1.0)`
:   CoopQ Linear layer with per-channel asymmetric quantization and STE backpropagation.
    Supports both standard nn.Linear and Conv1D weight representations.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, bits: int = 4, symmetric: bool = False, per_channel: bool = True, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.coopq_quantizer.CoopQLinear`
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

`CoopQQuantizer(model: nn.Module, bits: int = 4, min_bits: int = 4, max_bits: int = 8, symmetric: bool = False, per_channel: bool = True, use_ste: bool = True, recalibrate_ln: bool = True, target_attention_only: bool = False, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   CoopQ: Cooperative Game-Theoretic Bit Allocation Quantization.
    Allocates discrete bit-widths (>= 4 bits) across transformer layers
    proportional to layer sensitivity (Frobenius energy / variance), guaranteeing
    a target mean bit-budget under strict distortion minimization.
    
    Args:
        model: PyTorch model.
        bits: Target average bit-width (default 4).
        min_bits: Minimum allowable bit-width per layer (floor enforced at >= 4).
        max_bits: Maximum allowable bit-width per layer (default 8).
        symmetric: Whether to use symmetric quantization (default False for asymmetric calibration).
        per_channel: Whether to use per-channel scale factors (default True).
        use_ste: Whether to substitute nn.Linear with CoopQLinear for QAT (default True).
        target_attention_only: If True, only quantize attention projection layers (skip FFN/MLP).
        cfg: Optional configuration dictionary, namespace, or object.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Quantize linear layers with layer-specific bit allocation (>= 4 bits) and per-channel asymmetric calibration.

`CoopQSTEFakeQuantize(*args, **kwargs)`
:   Straight-Through Estimator (STE) fake-quantization autograd function for CoopQ.
    Forwards per-channel quantized-dequantized weights and passes gradients through on backward pass.

    ### Ancestors (in MRO)

    * torch.autograd.function.Function
    * torch.autograd.function._SingleLevelFunction
    * torch._C._FunctionBase
    * torch.autograd.function.FunctionCtx
    * torch.autograd.function._HookMixin

    ### Static methods

    `backward(ctx: Any, grad_output: torch.Tensor) ‑> tuple[torch.Tensor, None, None, None, None]`
    :   Define a formula for differentiating the operation with backward mode automatic differentiation.
        
        This function is to be overridden by all subclasses.
        (Defining this function is equivalent to defining the ``vjp`` function.)
        
        It must accept a context :attr:`ctx` as the first argument, followed by
        as many outputs as the :func:`forward` returned (None will be passed in
        for non tensor outputs of the forward function),
        and it should return as many tensors, as there were inputs to
        :func:`forward`. Each argument is the gradient w.r.t the given output,
        and each returned value should be the gradient w.r.t. the
        corresponding input. If an input is not a Tensor or is a Tensor not
        requiring grads, you can just pass None as a gradient for that input.
        
        The context can be used to retrieve tensors saved during the forward
        pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
        of booleans representing whether each input needs gradient. E.g.,
        :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
        first input to :func:`forward` needs gradient computed w.r.t. the
        output.

    `forward(ctx: Any, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, qmin: float, qmax: float) ‑> torch.Tensor`
    :   Define the forward of the custom autograd Function.
        
        This function is to be overridden by all subclasses.
        There are two ways to define forward:
        
        Usage 1 (Combined forward and ctx)::
        
            @staticmethod
            def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
                pass
        
        - It must accept a context ctx as the first argument, followed by any
          number of arguments (tensors or other types).
        - See :ref:`combining-forward-context` for more details
        
        Usage 2 (Separate forward and ctx)::
        
            @staticmethod
            def forward(*args: Any, **kwargs: Any) -> Any:
                pass
        
        
            @staticmethod
            def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
                pass
        
        - The forward no longer accepts a ctx argument.
        - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
          staticmethod to handle setting up the ``ctx`` object.
          ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
          to the forward.
        - See :ref:`extending-autograd` for more details
        
        The context can be used to store arbitrary data that can be then
        retrieved during the backward pass. Tensors should not be stored
        directly on `ctx` (though this is not currently enforced for
        backward compatibility). Instead, tensors should be saved either with
        :func:`ctx.save_for_backward` if they are intended to be used in
        ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
        if they are intended to be used for in ``jvp``.

`DRONECompressor(model: nn.Module, tokenizer: Any | None = None, rank: int = 64, max_error: float = 0.5, fallback_to_original: bool = True, device: str | torch.device | None = None, target_modules: list[str] | None = None, target_attention_only: bool = True, **kwargs: Any)`
:   DRONE Compressor.
    Data-aware SVD with identity weighting (standard data-independent SVD baseline)
    using identical rank adaptation, parameter constraints, and fallback error mechanism as AASVD.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress_linear(self, layer: nn.Module, rank: int | None = None) ‑> tuple[neuralops.compression_engine.aa_svd_compressor.SVDLinear | None, float, int]`
    :

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

`DirectSVDCompressor(model: nn.Module, rank: int | None = 32, min_rank: int = 16, rank_ratio: float | None = None, max_error: float = 0.35, fallback_to_original: bool = True, protect_critical: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = True, target_attention_only: bool = False, max_retries: int = 2, target_modules: list[str] | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   Direct SVD Compressor.
        
        Performs direct truncated SVD on target linear and Conv1D layers:
            W ≈ U_r * Sigma_r * V_r^T
        Allocates singular energy symmetrically:
            U_factor = U_r * sqrt(Sigma_r)
            V_factor = sqrt(Sigma_r) * V_r^T
        and replaces dense layers with factorized SVDLinear modules.
        
        Production hardening features:
        - Relative Frobenius reconstruction error validation with configurable max_error.
        - Automatic rank escalation retries when error threshold is exceeded.
        - Safe fallback to original layer if error remains high or decomposition fails.
        - Fallback and non-target layers frozen (requires_grad=False) during recovery.
        - Robust LoRALinear and Conv1D support (with transpose and adapter preservation).
        - target_attention_only parameter: restricts SVD factorization to attention projections.
        - Strict rank safeguard: enforces r < min(in_f, out_f) and parameter reduction r * (in + out) < in * out.
        - Unified critical-layer protection via should_compress_linear.
        - Full FP32 computation for numerical stability.
        - Real metrics recording: original_layer_count, compressed_layer_count, fallback_layer_count, effective_parameter_ratio.

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress target linear and Conv1D layers across the model.

    `compress_linear(self, layer: nn.Module, rank: int | None = None) ‑> tuple[neuralops.compression_engine.aa_svd_compressor.SVDLinear | None, float, int]`
    :   Decompose a single linear or Conv1D layer via SVD with symmetric factor distribution.
        Returns: (svd_layer, relative_reconstruction_error, chosen_rank)

`DirectSVDEncoderLayer(config: Any | None = None, hidden_size: int | None = None, intermediate_size: int | None = None, num_attention_heads: int | None = None, svd_rank: int = 20, layer_norm_eps: float = 1e-12, hidden_dropout_prob: float = 0.1, lora_r: int = 0, lora_alpha: float = 1.0)`
:   Encoder layer that replaces MHA with SVDSingleHead + standard FFN.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, output_attentions: bool = False, *args: Any, **kwargs: Any) ‑> torch.Tensor | tuple[torch.Tensor, ...]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

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

`DynamicEntropySVDCompressorRegistered(model: nn.Module, min_rank: int = 16, max_rank: int = 128, target_param_ratio: float | None = None, total_rank_budget: int | None = None, target_rank: int | None = None, max_error: float = 0.35, target_attention_only: bool = False, fallback_to_original: bool = True, calibration_loader: Any = None, calibration_texts: list[str] | None = None, target_modules: list[str] | None = None, tokenizer: Any | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Dynamic Entropy SVD Compressor.
    
    1. Computes singular entropy H(Sigma_l) on layer activation covariances or weights.
    2. Allocates layer ranks within [min_rank, max_rank] subject to parameter/rank budget.
    3. Replaces dense Linear/Conv1D layers with DynamicSVDLinear factorized modules.
    4. Provides relative Frobenius error validation, max_error threshold, and safe fallback.
    5. Freezes fallback and non-target layers during recovery.

    ### Ancestors (in MRO)

    * neuralops.compression_engine.dynamic_rank.DynamicEntropySVDCompressor
    * neuralops.core.base.BaseCompressor

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

`ENLightenCompressor(model: nn.Module, rank: int = 64, sparsity: float = 0.5, structured_24: bool = True, max_error: float = 0.45, target_attention_only: bool = False, fallback_to_original: bool = True, max_retries: int = 2, target_modules: list[str] | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   ENLighten Compressor: Low-Rank SVD + 2:4 Structured Sparsity.
        Decomposes weights into low-rank factors plus 2:4 structured sparse residuals,
        preserving Frobenius energy while reducing memory bandwidth and arithmetic complexity.
        
        Production hardening features:
        - Robust LoRALinear wrapper and Conv1D support (with transpose and adapter preservation).
        - target_attention_only parameter: restricts compression to attention projections and freezes non-target layers.
        - Relative Frobenius reconstruction error validation with configurable max_error.
        - Automatic rank escalation retries when error threshold is exceeded.
        - Safe fallback to original module if error remains high or decomposition is unstable, with fallback freezing.
        - Fully vectorized 2:4 structured pruning eliminating Python loop overhead.
        - Critical-layer protection via should_compress_linear.
        - Real compression metrics recording (effective_parameter_ratio, num_compressed, num_fallback, etc.).

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress linear and Conv1D layers using low-rank SVD + 2:4 structured sparse residuals.

`ERCCompressor(model: nn.Module, allowed_error: float = 0.2, max_error: float | None = None, fallback_to_original: bool = True, target_attention_only: bool = False, rank: int | None = None, max_retries: int = 2, target_modules: list[str] | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   Error-Controlled SVD (ERC-SVD) Compressor.
        
        Finds the minimal rank r per layer such that
            ||W - W_r||_F / ||W||_F <= allowed_error.
        Factored symmetrically (U_r sqrt(S), sqrt(S) V_r^T) and replaces with SVDLinear.

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress all target linear and Conv1D layers across the model.

    `compress_linear(self, module: nn.Module) ‑> tuple[neuralops.compression_engine.aa_svd_compressor.SVDLinear | None, float, int]`
    :   Decompose a single linear, Conv1D, or LoRALinear layer via error-controlled SVD.

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

`EntropyRankAllocatorRegistered(model: nn.Module, min_rank: int = 16, max_rank: int = 128, target_param_ratio: float | None = None, total_rank_budget: int | None = None, target_rank: int | None = None, max_error: float = 0.35, target_attention_only: bool = False, fallback_to_original: bool = True, calibration_loader: Any = None, calibration_texts: list[str] | None = None, target_modules: list[str] | None = None, tokenizer: Any | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Dynamic Entropy SVD Compressor.
    
    1. Computes singular entropy H(Sigma_l) on layer activation covariances or weights.
    2. Allocates layer ranks within [min_rank, max_rank] subject to parameter/rank budget.
    3. Replaces dense Linear/Conv1D layers with DynamicSVDLinear factorized modules.
    4. Provides relative Frobenius error validation, max_error threshold, and safe fallback.
    5. Freezes fallback and non-target layers during recovery.

    ### Ancestors (in MRO)

    * neuralops.compression_engine.dynamic_rank.DynamicEntropySVDCompressor
    * neuralops.core.base.BaseCompressor

`FAQLinear(in_features: int, out_features: int, bias: bool = True, bits: int = 8, symmetric: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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
    :   Linear with FAQ activation-aware quantization, STE backprop and optional LoRA.

    ### Static methods

    `from_linear(module: nn.Linear | Any, act_scale: torch.Tensor | None = None, bits: int = 8, symmetric: bool = True, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.faq_quantizer.FAQLinear`
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

    `get_weight(self) ‑> torch.Tensor`
    :

`FAQQuantizer(model: nn.Module, tokenizer: Any | None = None, bits: int = 8, calibration_texts: list[str] | None = None, calibration_data: Any | None = None, alpha: float = 0.5, symmetric: bool = True, use_ste: bool = True, lora_r: int = 16, lora_alpha: float = 1.0, max_error: float = 0.35, target_attention_only: bool = False, fallback_to_original: bool = True, skip_attention_qkv: bool = False, cfg: Any | None = None, num_calibration_batches: int = 8, **kwargs: Any)`
:   FAQQuantizer: Feature-Activation-Aware Quantization.
    
    Collects activation statistics to scale channel-wise quantization step sizes,
    minimizing activation-weighted reconstruction distortion.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

`FLAPCompressor(model: nn.Module, tokenizer: Any | None = None, sparsity: float = 0.2, metric_type: str = 'WIFV', max_error: float = 0.35, target_attention_only: bool = False, fallback_to_original: bool = True, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, structured_24: bool = False, target_modules: list[str] | None = None, cfg: Any | None = None, num_calibration_batches: int = 8, **kwargs: Any)`
:   FLAP Compressor: Fluctuation-based Adaptive Structured Pruning.
    
    Production hardening features:
    - Robust LoRALinear and Conv1D support (transpose + adapter preservation).
    - target_attention_only restricts pruning to attention projections.
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - Safe fallback to original module if error exceeds threshold.
    - Real metrics (effective_parameter_ratio, num_compressed, num_fallback, sparsity).

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

`FLATLLMCompressor(model: nn.Module, tokenizer: Any | None = None, keep_ratio: float = 0.7, damping: float = 0.0001, max_error: float = 0.35, target_attention_only: bool = False, fallback_to_original: bool = True, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, structured_24: bool = False, target_modules: list[str] | None = None, cfg: Any | None = None, num_calibration_batches: int = 8, **kwargs: Any)`
:   FLAT-LLM compressor: Activation-aware output-channel pruning.
    
    Pipeline:
      1. Collect per-output-channel energy from calibration activations.
      2. Rank channels by energy.
      3. Keep top-k channels (k = round(keep_ratio * out_features)).
      4. Replace Linear with PrunedChannelLinear (factorized storage).
    
    Optionally applies 2:4 structured sparsity (compute-only; no storage savings
    unless a sparse kernel is used).

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`FRanDICompressor(model: nn.Module | None = None, rank: int = 8, energy_threshold: float | None = None, calibration_loader: Any | None = None, lora_r: int = 0, lora_alpha: float = 1.0, max_error: float = 0.5, target_attention_only: bool = False, fallback_to_original: bool = True, skip_attention_qkv: bool = False, cfg: Any | None = None, target_layers: list[str] | None = None, num_calibration_batches: int = 16, **kwargs: Any)`
:   FRanDI (Feature-Regression aligned SVD) Compressor.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`FRanDILinear(in_features: int, out_features: int, rank: int = 8, bias: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Feature-Regression Aligned Factorized Linear Layer with optional LoRA.
    
    W ≈ B @ A
      - A (rank, in_features): input feature-aligned basis
      - B (out_features, rank): output expansion matrix
    
    Forward:
      y = (x @ A^T) @ B^T + bias
          + (alpha/r) * (x @ lora_A^T @ lora_B^T)
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_linear(linear: nn.Linear | Any, rank: int = 8, energy_threshold: float | None = None, cov_matrix: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.frandi_compressor.FRanDILinear`
    :

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

    `get_weight(self) ‑> torch.Tensor`
    :

    `merge(self) ‑> None`
    :   Fold LoRA delta into base factors (approximate, additive).

    `merge_and_unload(self) ‑> neuralops.compression_engine.frandi_compressor.FRanDILinear`
    :

    `unmerge(self) ‑> None`
    :

`FWSVDCompressor(model: nn.Module, tokenizer: Any | None = None, rank: int = 64, max_error: float = 0.5, fallback_to_original: bool = True, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, target_modules: list[str] | None = None, target_attention_only: bool = True, **kwargs: Any)`
:   Fisher-Weighted SVD Compressor.
    Projects weights using diagonal Fisher activation weights F_i = E[x_i^2].
    Replaces dense nn.Linear with SVDLinear.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress_linear(self, layer: nn.Module, rank: int | None = None, fisher_diag: torch.Tensor | None = None) ‑> tuple[neuralops.compression_engine.aa_svd_compressor.SVDLinear | None, float, int]`
    :

`FusedQKVSVDLinear(in_features: int, out_features: int, rank: int, bias: bool = True, dtype: torch.dtype | None = None, device: torch.device | None = None, lora_r: int = 0, lora_alpha: float = 1.0)`
:   Compressed replacement for GPT-2's fused c_attn.
    
    Stores three independent low-rank factors (Q, K, V). Forward concatenates
    them so the output shape matches the original fused projection.
    
    Supports mean-shift bias absorption: if `_input_mean` is set, the input
    is centered before the low-rank projection and the shift is added back
    through the bias.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstruct the fused (in, out) Conv1D weight for compatibility.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `set_fused_factors(self, W_fused: torch.Tensor) ‑> None`
    :   Split a fused (in, out) or (out, in) weight into Q/K/V factors.

    `set_input_mean(self, mu: torch.Tensor) ‑> None`
    :   Register the calibration activation mean (input centering).

`FusedSVDLinear(in_features: int, out_features: int, rank: int = 32, bias: bool = True, use_rms_norm: bool = False, eps: float = 1e-06, dtype: torch.dtype = torch.float32, device: torch.device | None = None)`
:   RESEARCH SOTA: Fused SVD-Linear Module.
    Combines low-rank projection $Z = x V^T$, in-SRAM RMSNorm, and up-projection $Y = Z U^T + b$
    into a single kernel invocation, eliminating HBM roundtrips.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix for compatibility.

    ### Methods

    `decompose_from_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) ‑> None`
    :   Decompose dense weight matrix into symmetric SVD factors.

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

    `get_orthogonality_loss(self) ‑> torch.Tensor`
    :   # RESEARCH FIX: Computes Stiefel manifold penalty to prevent singular subspace collapse.

    `reset_parameters(self) ‑> None`
    :

    `set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) ‑> None`
    :   Set precomputed U and V low-rank factors.

`GFWSCompressor(model: nn.Module, rank: int = 64, damping: float = 0.01, max_error: float = 0.2, fallback_to_original: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = True, target_attention_only: bool = False, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, target_modules: list[str] | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   Generalized Fisher / Covariance-Weighted SVD Compressor.
        Whitens weights using empirical input activation covariance square-root Cov(X)^{1/2},
        performs low-rank truncated SVD, and de-whitens to restore optimal compressed weights.
        
        Production hardening features:
        - Relative Frobenius reconstruction error validation with configurable max_error.
        - Automatic fallback to standard weight SVD if covariance whitening fails or error > max_error.
        - Safe fallback to original weights if even standard SVD exceeds max_error.
        - Robust Conv1D support (GPT-2 c_attn, c_proj) via weight transposition.
        - Unified critical-layer protection via should_compress_linear.
        - Full FP32 computation for numerical stability, casting back to original dtype only when assigning.
        - Trainable LoRA adapters (lora_A, lora_B) for parameter-efficient fine-tuning.
        - Per-layer logging and detailed compression statistics.

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress layers via activation covariance whitening + low-rank SVD, with standard SVD fallback.

`GFWSLoRALinear(in_features: int, out_features: int, weight: torch.Tensor, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Linear layer containing Fisher/Covariance-whitened SVD compressed base weights (frozen)
    with optional trainable LoRA adapters (lora_A, lora_B).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, weight: torch.Tensor, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.gfws_d_compressor.GFWSLoRALinear`
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

`GreCoDeCCompressor(model: nn.Module | None = None, rank: int = 16, max_iter: int = 15, lora_r: int = 0, lora_alpha: float = 1.0, max_error: float = 0.35, target_attention_only: bool = False, fallback_to_original: bool = True, skip_attention_qkv: bool = False, cfg: Any | None = None, target_layers: list[str] | None = None, **kwargs: Any)`
:   GreCoDeC Compressor.
    Decomposes linear layers using coordinate descent low-rank factorization.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, model: nn.Module | None = None) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress linear layers using GreCoDeC decomposition.

`GreCoDeCLinear(in_features: int, out_features: int, rank: int = 16, bias: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, dtype: torch.dtype | None = None, device: torch.device | None = None)`
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
    :   Greedy Coordinate Descent Factorized Linear Layer with optional LoRA adapter.
        W ≈ U @ V
        - U in R^{out_features x rank}
        - V in R^{rank x in_features}
        
        Forward pass:
        y = ((x @ V^T) @ U^T) + bias + (x @ lora_A^T @ lora_B^T) * (alpha / r)

    ### Static methods

    `from_linear(linear: nn.Linear | Any, rank: int = 16, max_iter: int = 15, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.grecodec_compressor.GreCoDeCLinear`
    :   Construct GreCoDeCLinear by greedy coordinate descent low-rank factorization.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix for compatibility.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Sub-quadratic forward execution.
        y = ((x @ V^T) @ U^T) + bias + LoRA

    `get_weight(self) ‑> torch.Tensor`
    :   Materialize full reconstructed weight matrix W = U @ V.

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

`HuffmanCoder()`
:   Vectorized lossless entropy coder for quantized weight and KV cache tensors.
    Supports continuous tensor quantization (e.g. 4-bit / 8-bit), symbol frequency modeling,
    binary prefix tree construction, numpy bitstream packing, and vectorized decoding.

    ### Static methods

    `build_tree(frequencies: dict[int, int]) ‑> neuralops.compression_engine.huffman_coder.HuffmanNode | None`
    :   Build Huffman binary prefix tree from symbol frequencies.

    `decode(packed_bytes: bytes, code_table: dict[int, str], metadata: dict[str, Any], device: torch.device | None = None) ‑> torch.Tensor`
    :   Vectorized decode bitstream back into reconstructed tensor.

    `encode(tensor: torch.Tensor, quantize_bits: int | None = 8) ‑> tuple[bytes, dict[int, str], dict[str, typing.Any]]`
    :   Vectorized encode input tensor into packed bitstream bytes and Huffman codebook.

    `generate_code_table(root: HuffmanNode | None, current_code: str = '', code_table: dict[int, str] | None = None) ‑> dict[int, str]`
    :   Traverse Huffman tree to generate prefix bitstring per symbol.

`HuffmanCompressor(model: nn.Module, quantize_bits: int = 8, max_error: float = 0.2, target_attention_only: bool = False, fallback_to_original: bool = True, cfg: Any | None = None, **kwargs: Any)`
:   HuffmanCompressor: Model-wide entropy compression profiler and simulator.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Profile and apply Huffman entropy coding across all linear layers.

`HuffmanNode(symbol: int | None = None, freq: int = 0, left: HuffmanNode | None = None, right: HuffmanNode | None = None)`
:   Node in the Huffman prefix tree.

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

`HybridSVDCompressor(model: nn.Module, svd_rank: int = 128, use_full_layer_replacement: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, max_error: float = 0.85, target_attention_only: bool = False, fallback_to_original: bool = True, cfg: Any | None = None, target_modules: list[str] | None = None, **kwargs: Any)`
:   Weight-based SVD compressor for Linear / Conv1D layers.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Decompose linear layers via SVD and replace with SVDLinear.

`KMeans(n_clusters=8, *, init='k-means++', n_init='auto', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd')`
:   K-Means clustering.
    
    Read more in the :ref:`User Guide <k_means>`.
    
    Parameters
    ----------
    
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.
    
        For an example of how to choose an optimal value for `n_clusters` refer to
        :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_silhouette_analysis.py`.
    
    init : {'k-means++', 'random'}, callable or array-like of shape             (n_clusters, n_features), default='k-means++'
        Method for initialization:
    
        * 'k-means++' : selects initial cluster centroids using sampling             based on an empirical probability distribution of the points'             contribution to the overall inertia. This technique speeds up             convergence. The algorithm implemented is "greedy k-means++". It             differs from the vanilla k-means++ by making several trials at             each sampling step and choosing the best centroid among them.
    
        * 'random': choose `n_clusters` observations (rows) at random from         data for the initial centroids.
    
        * If an array is passed, it should be of shape (n_clusters, n_features)        and gives the initial centers.
    
        * If a callable is passed, it should take arguments X, n_clusters and a        random state and return an initialization.
    
        For an example of how to use the different `init` strategies, see
        :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_digits.py`.
    
        For an evaluation of the impact of initialization, see the example
        :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_stability_low_dim_dense.py`.
    
    n_init : 'auto' or int, default='auto'
        Number of times the k-means algorithm is run with different centroid
        seeds. The final results is the best output of `n_init` consecutive runs
        in terms of inertia. Several runs are recommended for sparse
        high-dimensional problems (see :ref:`kmeans_sparse_high_dim`).
    
        When `n_init='auto'`, the number of runs depends on the value of init:
        10 if using `init='random'` or `init` is a callable;
        1 if using `init='k-means++'` or `init` is an array-like.
    
        .. versionadded:: 1.2
           Added 'auto' option for `n_init`.
    
        .. versionchanged:: 1.4
           Default value for `n_init` changed to `'auto'`.
    
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.
    
    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    
    verbose : int, default=0
        Verbosity mode.
    
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    
    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.
    
    algorithm : {"lloyd", "elkan"}, default="lloyd"
        K-means algorithm to use. The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However it's
        more memory intensive due to the allocation of an extra array of shape
        `(n_samples, n_clusters)`.
    
        .. versionchanged:: 0.18
            Added Elkan algorithm
    
        .. versionchanged:: 1.1
            Renamed "full" to "lloyd", and deprecated "auto" and "full".
            Changed "auto" to use "lloyd" instead of "elkan".
    
    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.
    
    labels_ : ndarray of shape (n_samples,)
        Labels of each point
    
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center,
        weighted by the sample weights if provided.
    
    n_iter_ : int
        Number of iterations run.
    
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    
        .. versionadded:: 0.24
    
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    
        .. versionadded:: 1.0
    
    See Also
    --------
    MiniBatchKMeans : Alternative online implementation that does incremental
        updates of the centers positions using mini-batches.
        For large scale learning (say n_samples > 10k) MiniBatchKMeans is
        probably much faster than the default batch implementation.
    
    Notes
    -----
    The k-means problem is solved using either Lloyd's or Elkan's algorithm.
    
    The average complexity is given by O(k n T), where n is the number of
    samples and T is the number of iteration.
    
    The worst case complexity is given by O(n^(k+2/p)) with
    n = n_samples, p = n_features.
    Refer to :doi:`"How slow is the k-means method?" D. Arthur and S. Vassilvitskii -
    SoCG2006.<10.1145/1137856.1137880>` for more details.
    
    In practice, the k-means algorithm is very fast (one of the fastest
    clustering algorithms available), but it falls in local minima. That's why
    it can be useful to restart it several times.
    
    If the algorithm stops before fully converging (because of ``tol`` or
    ``max_iter``), ``labels_`` and ``cluster_centers_`` will not be consistent,
    i.e. the ``cluster_centers_`` will not be the means of the points in each
    cluster. Also, the estimator will reassign ``labels_`` after the last
    iteration to make ``labels_`` consistent with ``predict`` on the training
    set.
    
    Examples
    --------
    
    >>> from sklearn.cluster import KMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    >>> kmeans.labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> kmeans.predict([[0, 0], [12, 3]])
    array([1, 0], dtype=int32)
    >>> kmeans.cluster_centers_
    array([[10.,  2.],
           [ 1.,  2.]])
    
    For examples of common problems with K-Means and how to address them see
    :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_assumptions.py`.
    
    For a demonstration of how K-Means can be used to cluster text documents see
    :ref:`sphx_glr_auto_examples_text_plot_document_clustering.py`.
    
    For a comparison between K-Means and MiniBatchKMeans refer to example
    :ref:`sphx_glr_auto_examples_cluster_plot_mini_batch_kmeans.py`.
    
    For a comparison between K-Means and BisectingKMeans refer to example
    :ref:`sphx_glr_auto_examples_cluster_plot_bisect_kmeans.py`.

    ### Ancestors (in MRO)

    * sklearn.cluster._kmeans._BaseKMeans
    * sklearn.base.ClassNamePrefixFeaturesOutMixin
    * sklearn.base.TransformerMixin
    * sklearn.utils._set_output._SetOutputMixin
    * sklearn.base.ClusterMixin
    * sklearn.base.BaseEstimator
    * sklearn.utils._repr_html.base.ReprHTMLMixin
    * sklearn.utils._repr_html.base._HTMLDocumentationLinkMixin
    * sklearn.utils._metadata_requests._MetadataRequester
    * abc.ABC

    ### Methods

    `fit(self, X, y=None, sample_weight=None)`
    :   Compute k-means clustering.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.
        
        y : Ignored
            Not used, present here for API consistency by convention.
        
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.
        
            .. versionadded:: 0.20
        
        Returns
        -------
        self : object
            Fitted estimator.

    `set_fit_request(self: sklearn.cluster._kmeans.KMeans, *, sample_weight: bool | str | None = '$UNCHANGED$') ‑> sklearn.cluster._kmeans.KMeans`
    :   Configure whether metadata should be requested to be passed to the ``fit`` method.
        
        Note that this method is only relevant when this estimator is used as a
        sub-estimator within a :term:`meta-estimator` and metadata routing is enabled
        with ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
        Please check the :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.
        
        The options for each parameter are:
        
        - ``True``: metadata is requested, and passed to ``fit`` if provided. The request is ignored if metadata is not provided.
        
        - ``False``: metadata is not requested and the meta-estimator will not pass it to ``fit``.
        
        - ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
        
        - ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.
        
        The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
        existing request. This allows you to change the request for some
        parameters and not others.
        
        .. versionadded:: 1.3
        
        Parameters
        ----------
        sample_weight : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED
            Metadata routing for ``sample_weight`` parameter in ``fit``.
        
        Returns
        -------
        self : object
            The updated object.

    `set_score_request(self: sklearn.cluster._kmeans.KMeans, *, sample_weight: bool | str | None = '$UNCHANGED$') ‑> sklearn.cluster._kmeans.KMeans`
    :   Configure whether metadata should be requested to be passed to the ``score`` method.
        
        Note that this method is only relevant when this estimator is used as a
        sub-estimator within a :term:`meta-estimator` and metadata routing is enabled
        with ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
        Please check the :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.
        
        The options for each parameter are:
        
        - ``True``: metadata is requested, and passed to ``score`` if provided. The request is ignored if metadata is not provided.
        
        - ``False``: metadata is not requested and the meta-estimator will not pass it to ``score``.
        
        - ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
        
        - ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.
        
        The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
        existing request. This allows you to change the request for some
        parameters and not others.
        
        .. versionadded:: 1.3
        
        Parameters
        ----------
        sample_weight : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED
            Metadata routing for ``sample_weight`` parameter in ``score``.
        
        Returns
        -------
        self : object
            The updated object.

`KMeansCompressor(model: nn.Module | None = None, n_clusters: int = 256, bits: int | None = None, max_error: float = 0.35, target_attention_only: bool = False, fallback_to_original: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = False, skip_should_compress_filter: bool = False, min_layer_dim: int = 8, kmeans_seed: int = 51966, cfg: Any | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`KMeansLinear(linear: nn.Module, n_clusters: int = 256, lora_r: int = 0, lora_alpha: float = 1.0, kmeans_seed: int = 51966)`
:   Linear layer with weight matrix quantized via 1D k-means codebook.
    
    Storage:
      - centroids:  (C,)         fp16/fp32
      - indices:    (out, in)    int8/int16/int32  (smallest that fits C)
      - bias:       (out,)       fp16/fp32
      - lora_A/B:   (r, in) / (out, r)   optional
    
    Forward:
      W_hat = centroids[indices]
      y = F.linear(x, W_hat) + bias + scaling · (x @ A^T) @ B^T
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `is_custom_linear: bool`
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

    `get_weight(self) ‑> torch.Tensor`
    :

    `storage_bytes(self) ‑> dict[str, int]`
    :   KM-FIX-07/22: honest storage accounting.

    `train(self, mode: bool = True) ‑> neuralops.compression_engine.kmeans_quantize.KMeansLinear`
    :   Set the module in training mode.
        
        This has an effect only on certain modules. See the documentation of
        particular modules for details of their behaviors in training/evaluation
        mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.
        
        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.
        
        Returns:
            Module: self

`KMeansSTEFunction(*args, **kwargs)`
:   STE for codebook lookup.
    
    Forward:  out = F.linear(x, centroids[indices], bias)
    Backward: gradient to centroids via scatter-add; gradient to x flows
              through the same effective W.

    ### Ancestors (in MRO)

    * torch.autograd.function.Function
    * torch.autograd.function._SingleLevelFunction
    * torch._C._FunctionBase
    * torch.autograd.function.FunctionCtx
    * torch.autograd.function._HookMixin

    ### Static methods

    `backward(ctx, grad_output)`
    :   Define a formula for differentiating the operation with backward mode automatic differentiation.
        
        This function is to be overridden by all subclasses.
        (Defining this function is equivalent to defining the ``vjp`` function.)
        
        It must accept a context :attr:`ctx` as the first argument, followed by
        as many outputs as the :func:`forward` returned (None will be passed in
        for non tensor outputs of the forward function),
        and it should return as many tensors, as there were inputs to
        :func:`forward`. Each argument is the gradient w.r.t the given output,
        and each returned value should be the gradient w.r.t. the
        corresponding input. If an input is not a Tensor or is a Tensor not
        requiring grads, you can just pass None as a gradient for that input.
        
        The context can be used to retrieve tensors saved during the forward
        pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
        of booleans representing whether each input needs gradient. E.g.,
        :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
        first input to :func:`forward` needs gradient computed w.r.t. the
        output.

    `forward(ctx, centroids, indices, x, bias)`
    :   Define the forward of the custom autograd Function.
        
        This function is to be overridden by all subclasses.
        There are two ways to define forward:
        
        Usage 1 (Combined forward and ctx)::
        
            @staticmethod
            def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
                pass
        
        - It must accept a context ctx as the first argument, followed by any
          number of arguments (tensors or other types).
        - See :ref:`combining-forward-context` for more details
        
        Usage 2 (Separate forward and ctx)::
        
            @staticmethod
            def forward(*args: Any, **kwargs: Any) -> Any:
                pass
        
        
            @staticmethod
            def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
                pass
        
        - The forward no longer accepts a ctx argument.
        - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
          staticmethod to handle setting up the ``ctx`` object.
          ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
          to the forward.
        - See :ref:`extending-autograd` for more details
        
        The context can be used to store arbitrary data that can be then
        retrieved during the backward pass. Tensors should not be stored
        directly on `ctx` (though this is not currently enforced for
        backward compatibility). Instead, tensors should be saved either with
        :func:`ctx.save_for_backward` if they are intended to be used in
        ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
        if they are intended to be used for in ``jvp``.

`KVTCCompressor(model: nn.Module, bits: int = 4, pca_energy: float = 0.95, max_error: float = 0.5, fallback_to_original: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, target_attention_only: bool = False, skip_attention_qkv: bool = True, min_rank: int = 4, target_modules: list[str] | None = None, **kwargs: Any)`
:   KVTC: KV Cache Transform Coding compressor (Static Weight PCA + Quantization).
    Compresses linear projection weights using PCA decorrelation,
    percentile clipping, uniform integer quantization, and entropy coding estimation.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress linear layers using PCA decorrelation and true integer quantization.

`KVTCQuantizedLoRALinear(in_features: int, out_features: int, int_weight: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, compute_dtype: torch.dtype = torch.float32)`
:   Linear layer containing true integer quantized base weights (int8/int16 buffer)
    with scale and zero_point buffers, dequantized on the fly during forward(),
    plus optional trainable LoRA adapters (lora_A, lora_B).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, int_weight: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, lora_r: int = 0, lora_alpha: float = 1.0, compute_dtype: torch.dtype | None = None) ‑> neuralops.compression_engine.kvtc_compressor.KVTCQuantizedLoRALinear`
    :

    ### Instance variables

    `weight: torch.Tensor`
    :   Backward-compatibility property returning on-the-fly dequantized weight.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`KVTCStore(bits: int = 4, pca_energy: float = 0.95)`
:   

    ### Class variables

    `is_custom_attention: bool`
    :   Runtime KV Cache Transform Coding engine.
        Applies PCA decorrelation across head feature dimensions and adaptive quantization,
        supporting dynamic sequence lengths across prefill and decoding phases.

    ### Methods

    `calibrate_basis(self, k: torch.Tensor, v: torch.Tensor) ‑> None`
    :   Calibrate PCA projection basis from sample K/V tensor (B, H, N, D).

    `compress_kv_cache(self, k: torch.Tensor, v: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor, dict[str, typing.Any]]`
    :   Transform and quantize dynamic KV cache tensors.
        Returns reconstructed (k_hat, v_hat) and coding statistics.

    `reset(self) ‑> None`
    :   Reset PCA basis to prevent cross-sequence memory leaks.

`KroneckerCompressor(model: nn.Module | None = None, rank: int | None = None, max_error: float = 0.8, fallback_to_original: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, target_attention_only: bool = False, skip_attention_qkv: bool = False, skip_should_compress_filter: bool = False, min_layer_dim: int = 8, cfg: Any | None = None, target_layers: list[str] | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`KroneckerLinear(in_features: int, out_features: int, bias: bool = True, rank: int | None = None, m1: int | None = None, n1: int | None = None, m2: int | None = None, n2: int | None = None, dtype: torch.dtype | None = None, device: torch.device | None = None, lora_r: int = 0, lora_alpha: float = 1.0)`
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
    :   The type of the None singleton.

    `is_custom_linear: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_linear(linear: nn.Module, rank: int | None = None, max_error: float = 0.2, m1: int | None = None, n1: int | None = None, m2: int | None = None, n2: int | None = None, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.kronecker_compressor.KroneckerLinear`
    :   Build a KroneckerLinear from a pretrained Linear/Conv1D/LoRA wrapper.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix (out_features, in_features).

    ### Methods

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

`KroneckerTTLoRACompressor(model: torch.nn.modules.module.Module | None = None, rank: int = 8, gamma: float = 1.0, lora_r: int = 4, lora_alpha: float = 8.0, max_error: float = 0.35, target_attention_only: bool = False, fallback_to_original: bool = True, skip_attention_qkv: bool = False, skip_should_compress_filter: bool = False, min_layer_dim: int = 8, cfg: Any | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress_linear(self, linear_layer: torch.nn.modules.module.Module) ‑> neuralops.compression_engine.kronecker_tt_lora.KroneckerTTLoRALayer`
    :

`KroneckerTTLoRALayer(base_layer: torch.nn.modules.linear.Linear, m1: int, n1: int, m2: int, n2: int, tt_shapes: tuple[int, int, int], tt_ranks: tuple[int, int, int, int] = (1, 4, 4, 1), gamma: float = 1.0, lora_r: int = 4, lora_alpha: float = 8.0, r_min: int = 2, r_max: int = 32, entropy_threshold: float = 0.75)`
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
    :   The type of the None singleton.

    `is_custom_linear: bool`
    :   The type of the None singleton.

    ### Static methods

    `factorize_dimensions(in_f: int, out_f: int) ‑> dict[str, typing.Any]`
    :   Auto-factorize dims and pick 3-factor TT shape.

    `from_linear(base_layer: torch.nn.modules.module.Module, m1: int | None = None, n1: int | None = None, m2: int | None = None, n2: int | None = None, tt_shapes: tuple[int, int, int] | None = None, tt_ranks: tuple[int, int, int, int] = (1, 4, 4, 1), gamma: float = 1.0, lora_r: int = 4, lora_alpha: float = 8.0, **kwargs: Any) ‑> neuralops.compression_engine.kronecker_tt_lora.KroneckerTTLoRALayer`
    :   Build a KroneckerTTLoRALayer from Linear / Conv1D / LoRA wrapper.

    ### Instance variables

    `bias: torch.Tensor | None`
    :

    `weight: torch.Tensor`
    :

    ### Methods

    `adapt_tt_ranks(self, delta_r: int = 1) ‑> None`
    :

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `get_delta_w(self) ‑> torch.Tensor`
    :

    `get_tt_tensor(self) ‑> torch.Tensor`
    :

`LIFTRCompressor(model: nn.Module, replace_attention: bool = False, gelu_coefficient: float = 0.8, validation_calibration: bool = False, calibration_threshold: float = 0.2, target_attention_only: bool = False, skip_should_compress_filter: bool = False, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `CRITICAL_MODULE_KEYWORDS`
    :   The type of the None singleton.

    `is_custom_attention: bool`
    :   The type of the None singleton.

`LSCCompressor(model: nn.Module, rank: int = 32, energy_ratio: float | None = None, max_error: float = 0.85, fallback_to_original: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = False, target_attention_only: bool = False, target_modules: list[str] | None = None, min_layer_dim: int = 8, skip_should_compress_filter: bool = False, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`LSCPrunedLoRALinear(in_features: int, out_features: int, weight: torch.Tensor, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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
    :   The type of the None singleton.

    `is_custom_linear: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_linear(linear: nn.Module, weight: torch.Tensor, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.lsc_compressor.LSCPrunedLoRALinear`
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

`LiLLaMACompressor(model: nn.Module, tokenizer: Any | None = None, rank: int = 32, calibration_texts: list[str] | None = None, calibration_loader: Any | None = None, device: str | torch.device | None = None, max_error: float = 0.2, fallback_to_original: bool = True, skip_attention: bool = False, target_attention_only: bool = False, target_modules: list[str] | None = None, min_layer_dim: int = 8, skip_should_compress_filter: bool = False, max_calibration_batches: int = 8, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `category: str`
    :   The type of the None singleton.

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Return (compressed_model, statistics).

`LiLLaMADistiller(model: nn.Module, tokenizer: Any | None = None, rank: int = 32, calibration_texts: list[str] | None = None, calibration_loader: Any | None = None, device: str | torch.device | None = None, max_error: float = 0.2, fallback_to_original: bool = True, skip_attention: bool = False, target_attention_only: bool = False, target_modules: list[str] | None = None, min_layer_dim: int = 8, skip_should_compress_filter: bool = False, max_calibration_batches: int = 8, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `category: str`
    :   The type of the None singleton.

    `is_custom_attention: bool`
    :   The type of the None singleton.

`LoRALinear(in_features: int, out_features: int, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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
    :   The type of the None singleton.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `freeze_base_weights(self) ‑> None`
    :

    `reset_parameters(self) ‑> None`
    :

    `unfreeze_base_weights(self) ‑> None`
    :

`LowRankCompressor(model: nn.Module, rank: int = 8, energy_threshold: float | None = None, max_error: float = 0.5, target_attention_only: bool = False, fallback_to_original: bool = True, target_modules: list[str] | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   SoKA Low-Rank Linear Compressor.
        Replaces dense nn.Linear and Conv1D layers with factorized LowRankLinear layers.

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Replaces target Linear and Conv1D modules with LowRankLinear.

`SoKACompressor(model: nn.Module, rank: int = 8, energy_threshold: float | None = None, max_error: float = 0.5, target_attention_only: bool = False, fallback_to_original: bool = True, target_modules: list[str] | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   SoKA Low-Rank Linear Compressor.
        Replaces dense nn.Linear and Conv1D layers with factorized LowRankLinear layers.

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Replaces target Linear and Conv1D modules with LowRankLinear.

`LowRankLinear(in_features: int, out_features: int, rank: int, bias: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Factorized Linear Layer executing as y = (x @ V) @ U^T + b.
    Reduces parameter storage from (d_in * d_out) to r * (d_in + d_out).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix for compatibility.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`MGIECompressor(model: nn.Module, rank: int | None = None, max_error: float = 0.5, fallback_to_original: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = False, target_attention_only: bool = False, min_rank: int = 4, target_modules: list[str] | None = None, **kwargs: Any)`
:   MGIE-SVD Compressor (Maximum Information Entropy SVD).

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`MGIESVDBudgetCompressor(model: nn.Module, total_rank_budget: int = 1000, rank: int | None = None, max_error: float = 0.5, fallback_to_original: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = False, target_attention_only: bool = False, min_rank: int = 4, target_modules: list[str] | None = None, **kwargs: Any)`
:   MGIE-SVD Budget Compressor: Global Entropy-Proportional Rank Allocation.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`MLoRQCompressor(model: nn.Module, total_memory_budget_mb: float = 500, bits_candidates: list[int] | None = None, max_error: float = 0.5, fallback_to_original: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = False, target_attention_only: bool = False, min_rank: int = 4, target_modules: list[str] | None = None, **kwargs: Any)`
:   MLoRQ Compressor: Multi-Layer Optimal Rank and Quantization under Memory Budget.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`MLoRQLoRALinear(in_features: int, out_features: int, rank: int, bits: int, U_packed: torch.Tensor, U_scale: torch.Tensor, U_shape: torch.Size, S_packed: torch.Tensor, S_scale: torch.Tensor, S_shape: torch.Size, Vt_packed: torch.Tensor, Vt_scale: torch.Tensor, Vt_shape: torch.Size, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Low-rank quantized linear layer:
    
        y = (x @ Vt) * S @ U^T + bias + LoRA(x)
    
    where U, S, Vt are stored as PACKED int8 tensors with per-tensor scales.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_linear(linear: nn.Module, U: torch.Tensor, S: torch.Tensor, Vt: torch.Tensor, bits: int, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.mlorq_compressor.MLoRQLoRALinear`
    :

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

    `storage_bytes(self) ‑> int`
    :

`MambaShedderCompressor(model: nn.Module, tokenizer: Any | None = None, sparsity: float = 0.2, shed_layers: int = 0, max_error: float = 0.2, fallback_to_original: bool = True, validate_after_shed: bool = True, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, structured_24: bool = False, target_attention_only: bool = False, target_modules: list[str] | None = None, max_calibration_batches: int = 8, min_layer_dim: int = 8, skip_should_compress_filter: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`MambaShedderPrunedLinear(in_features: int, original_out_features: int, W_kept: torch.Tensor, keep_idx: torch.Tensor, bias_kept: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0)`
:   Linear layer with pruned output channels.
    
    Stores only the kept rows of W:
      - W_kept:    (k, in_features)
      - bias_kept: (k,) or None
      - keep_idx:  (k,) int64 (output positions of kept rows)
      - original_out_features: int
    
    Forward:
      h = F.linear(x, W_kept, bias_kept)              # (..., k)
      out = zeros(..., original_out) then scatter to keep_idx
      + LoRA(x) if present.
    
    Storage: k * in instead of out * in.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstruct (original_out, in) weight with zeros in pruned rows.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `storage_bytes(self) ‑> int`
    :

`MiniBatchKMeans(n_clusters=8, *, init='k-means++', max_iter=100, batch_size=1024, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init='auto', reassignment_ratio=0.01)`
:   Mini-Batch K-Means clustering.
    
    Read more in the :ref:`User Guide <mini_batch_kmeans>`.
    
    Parameters
    ----------
    
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.
    
    init : {'k-means++', 'random'}, callable or array-like of shape             (n_clusters, n_features), default='k-means++'
        Method for initialization:
    
        'k-means++' : selects initial cluster centroids using sampling based on
        an empirical probability distribution of the points' contribution to the
        overall inertia. This technique speeds up convergence. The algorithm
        implemented is "greedy k-means++". It differs from the vanilla k-means++
        by making several trials at each sampling step and choosing the best centroid
        among them.
    
        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.
    
        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
    
        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.
    
        For an evaluation of the impact of initialization, see the example
        :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_stability_low_dim_dense.py`.
    
    max_iter : int, default=100
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.
    
    batch_size : int, default=1024
        Size of the mini batches.
        For faster computations, you can set `batch_size > 256 * number_of_cores`
        to enable :ref:`parallelism <lower-level-parallelism-with-openmp>`
        on all cores.
    
        .. versionchanged:: 1.0
           `batch_size` default changed from 100 to 1024.
    
    verbose : int, default=0
        Verbosity mode.
    
    compute_labels : bool, default=True
        Compute label assignment and inertia for the complete dataset
        once the minibatch optimization has converged in fit.
    
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    
    tol : float, default=0.0
        Control early stopping based on the relative center changes as
        measured by a smoothed, variance-normalized of the mean center
        squared position changes. This early stopping heuristics is
        closer to the one used for the batch variant of the algorithms
        but induces a slight computational and memory overhead over the
        inertia heuristic.
    
        To disable convergence detection based on normalized center
        change, set tol to 0.0 (default).
    
    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed inertia.
    
        To disable convergence detection based on inertia, set
        max_no_improvement to None.
    
    init_size : int, default=None
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than n_clusters.
    
        If `None`, the heuristic is `init_size = 3 * batch_size` if
        `3 * batch_size < n_clusters`, else `init_size = 3 * n_clusters`.
    
    n_init : 'auto' or int, default="auto"
        Number of random initializations that are tried.
        In contrast to KMeans, the algorithm is only run once, using the best of
        the `n_init` initializations as measured by inertia. Several runs are
        recommended for sparse high-dimensional problems (see
        :ref:`kmeans_sparse_high_dim`).
    
        When `n_init='auto'`, the number of runs depends on the value of init:
        3 if using `init='random'` or `init` is a callable;
        1 if using `init='k-means++'` or `init` is an array-like.
    
        .. versionadded:: 1.2
           Added 'auto' option for `n_init`.
    
        .. versionchanged:: 1.4
           Default value for `n_init` changed to `'auto'` in version.
    
    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a center to
        be reassigned. A higher value means that low count centers are more
        easily reassigned, which means that the model will take longer to
        converge, but should converge in a better clustering. However, too high
        a value may cause convergence issues, especially with a small batch
        size.
    
    Attributes
    ----------
    
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    
    labels_ : ndarray of shape (n_samples,)
        Labels of each point (if compute_labels is set to True).
    
    inertia_ : float
        The value of the inertia criterion associated with the chosen
        partition if compute_labels is set to True. If compute_labels is set to
        False, it's an approximation of the inertia based on an exponentially
        weighted average of the batch inertiae.
        The inertia is defined as the sum of square distances of samples to
        their cluster center, weighted by the sample weights if provided.
    
    n_iter_ : int
        Number of iterations over the full dataset.
    
    n_steps_ : int
        Number of minibatches processed.
    
        .. versionadded:: 1.0
    
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    
        .. versionadded:: 0.24
    
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    
        .. versionadded:: 1.0
    
    See Also
    --------
    KMeans : The classic implementation of the clustering method based on the
        Lloyd's algorithm. It consumes the whole set of input data at each
        iteration.
    
    Notes
    -----
    See https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf
    
    When there are too few points in the dataset, some centers may be
    duplicated, which means that a proper clustering in terms of the number
    of requesting clusters and the number of returned clusters will not
    always match. One solution is to set `reassignment_ratio=0`, which
    prevents reassignments of clusters that are too small.
    
    See :ref:`sphx_glr_auto_examples_cluster_plot_birch_vs_minibatchkmeans.py` for a
    comparison with :class:`~sklearn.cluster.BIRCH`.
    
    Examples
    --------
    >>> from sklearn.cluster import MiniBatchKMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 0], [4, 4],
    ...               [4, 5], [0, 1], [2, 2],
    ...               [3, 2], [5, 5], [1, -1]])
    >>> # manually fit on batches
    >>> kmeans = MiniBatchKMeans(n_clusters=2,
    ...                          random_state=0,
    ...                          batch_size=6,
    ...                          n_init="auto")
    >>> kmeans = kmeans.partial_fit(X[0:6,:])
    >>> kmeans = kmeans.partial_fit(X[6:12,:])
    >>> kmeans.cluster_centers_
    array([[3.375, 3.  ],
           [0.75 , 0.5 ]])
    >>> kmeans.predict([[0, 0], [4, 4]])
    array([1, 0], dtype=int32)
    >>> # fit on the whole data
    >>> kmeans = MiniBatchKMeans(n_clusters=2,
    ...                          random_state=0,
    ...                          batch_size=6,
    ...                          max_iter=10,
    ...                          n_init="auto").fit(X)
    >>> kmeans.cluster_centers_
    array([[3.55102041, 2.48979592],
           [1.06896552, 1.        ]])
    >>> kmeans.predict([[0, 0], [4, 4]])
    array([1, 0], dtype=int32)
    
    For a comparison of Mini-Batch K-Means clustering with other clustering algorithms,
    see :ref:`sphx_glr_auto_examples_cluster_plot_cluster_comparison.py`

    ### Ancestors (in MRO)

    * sklearn.cluster._kmeans._BaseKMeans
    * sklearn.base.ClassNamePrefixFeaturesOutMixin
    * sklearn.base.TransformerMixin
    * sklearn.utils._set_output._SetOutputMixin
    * sklearn.base.ClusterMixin
    * sklearn.base.BaseEstimator
    * sklearn.utils._repr_html.base.ReprHTMLMixin
    * sklearn.utils._repr_html.base._HTMLDocumentationLinkMixin
    * sklearn.utils._metadata_requests._MetadataRequester
    * abc.ABC

    ### Methods

    `fit(self, X, y=None, sample_weight=None)`
    :   Compute the centroids on X by chunking it into mini-batches.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory copy
            if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.
        
        y : Ignored
            Not used, present here for API consistency by convention.
        
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.
        
            .. versionadded:: 0.20
        
        Returns
        -------
        self : object
            Fitted estimator.

    `partial_fit(self, X, y=None, sample_weight=None)`
    :   Update k means estimate on a single mini-batch X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory copy
            if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.
        
        y : Ignored
            Not used, present here for API consistency by convention.
        
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.
        
        Returns
        -------
        self : object
            Return updated estimator.

    `set_fit_request(self: sklearn.cluster._kmeans.MiniBatchKMeans, *, sample_weight: bool | str | None = '$UNCHANGED$') ‑> sklearn.cluster._kmeans.MiniBatchKMeans`
    :   Configure whether metadata should be requested to be passed to the ``fit`` method.
        
        Note that this method is only relevant when this estimator is used as a
        sub-estimator within a :term:`meta-estimator` and metadata routing is enabled
        with ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
        Please check the :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.
        
        The options for each parameter are:
        
        - ``True``: metadata is requested, and passed to ``fit`` if provided. The request is ignored if metadata is not provided.
        
        - ``False``: metadata is not requested and the meta-estimator will not pass it to ``fit``.
        
        - ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
        
        - ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.
        
        The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
        existing request. This allows you to change the request for some
        parameters and not others.
        
        .. versionadded:: 1.3
        
        Parameters
        ----------
        sample_weight : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED
            Metadata routing for ``sample_weight`` parameter in ``fit``.
        
        Returns
        -------
        self : object
            The updated object.

    `set_partial_fit_request(self: sklearn.cluster._kmeans.MiniBatchKMeans, *, sample_weight: bool | str | None = '$UNCHANGED$') ‑> sklearn.cluster._kmeans.MiniBatchKMeans`
    :   Configure whether metadata should be requested to be passed to the ``partial_fit`` method.
        
        Note that this method is only relevant when this estimator is used as a
        sub-estimator within a :term:`meta-estimator` and metadata routing is enabled
        with ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
        Please check the :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.
        
        The options for each parameter are:
        
        - ``True``: metadata is requested, and passed to ``partial_fit`` if provided. The request is ignored if metadata is not provided.
        
        - ``False``: metadata is not requested and the meta-estimator will not pass it to ``partial_fit``.
        
        - ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
        
        - ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.
        
        The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
        existing request. This allows you to change the request for some
        parameters and not others.
        
        .. versionadded:: 1.3
        
        Parameters
        ----------
        sample_weight : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED
            Metadata routing for ``sample_weight`` parameter in ``partial_fit``.
        
        Returns
        -------
        self : object
            The updated object.

    `set_score_request(self: sklearn.cluster._kmeans.MiniBatchKMeans, *, sample_weight: bool | str | None = '$UNCHANGED$') ‑> sklearn.cluster._kmeans.MiniBatchKMeans`
    :   Configure whether metadata should be requested to be passed to the ``score`` method.
        
        Note that this method is only relevant when this estimator is used as a
        sub-estimator within a :term:`meta-estimator` and metadata routing is enabled
        with ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
        Please check the :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.
        
        The options for each parameter are:
        
        - ``True``: metadata is requested, and passed to ``score`` if provided. The request is ignored if metadata is not provided.
        
        - ``False``: metadata is not requested and the meta-estimator will not pass it to ``score``.
        
        - ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
        
        - ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.
        
        The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
        existing request. This allows you to change the request for some
        parameters and not others.
        
        .. versionadded:: 1.3
        
        Parameters
        ----------
        sample_weight : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED
            Metadata routing for ``sample_weight`` parameter in ``score``.
        
        Returns
        -------
        self : object
            The updated object.

`MoDeGPTCompressor(model: nn.Module, rank: int = 64, method: str = 'svd', max_error: float = 0.5, fallback_to_original: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = False, target_attention_only: bool = False, min_rank: int = 4, target_modules: list[str] | None = None, **kwargs: Any)`
:   MoDeGPT: Modular Decomposition of Transformer Projection Blocks.
    
    Shares a low-rank input basis V across paired projections sharing the same
    input dimension. This yields real storage reduction for the pair:
    
        Original: out1·in + out2·in        = (out1+out2)·in
        MoDeGPT : r·in + out1·r + out2·r   = r·(in + out1 + out2)
    
    Storage reduction when r < in·(out1+out2) / (in + out1 + out2).

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`MoDeGPTLowRankLinear(in_features: int, out_features: int, rank: int, U: torch.Tensor, V: torch.Tensor | None = None, shared_V: torch.Tensor | None = None, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Low-rank linear layer.
    
        y = x @ V^T @ U^T + bias
          = (x @ V^T) @ U^T + bias
        where U ∈ R^{out × r}, V ∈ R^{r × in}
    
    If `shared_V` is provided, this module uses the external tensor instead of
    its own `V` parameter. This enables sharing the input basis across paired
    projections (Q/K, Gate/Up, ...), giving real compression for the pair.
    
    Optional LoRA adapters (lora_A, lora_B) are trained in addition to the
    frozen low-rank base.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed (out, in) weight matrix.

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

`NLISoftmax(dim: int = -1)`
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
    :   The type of the None singleton.

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

`NLISoftmaxAttention(config: Any | None = None, rank: int | None = None, dim: int | None = None, num_heads: int | None = None, is_nested_encoder: bool = False, is_decoder: bool | None = None, **kwargs: Any)`
:   Attention approximating softmax with NLISoftmax.
    
    Nested-encoder mode (BERT/RoBERTa): the internal out_proj is NOT created,
    because the outer `BertSelfOutput` / `RobertaSelfOutput` applies dense + LayerNorm.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `key: torch.nn.modules.module.Module`
    :

    `query: torch.nn.modules.module.Module`
    :

    `value: torch.nn.modules.module.Module`
    :

    ### Methods

    `freeze_base_weights(self) ‑> None`
    :

`NLISoftmaxCompressor(model: torch.nn.modules.module.Module, max_error: float = 0.2, fallback_to_original: bool = True, target_attention_only: bool = False, skip_should_compress_filter: bool = False, cfg: Any | None = None, **kwargs: Any)`
:   Replaces nn.Softmax modules and nested self-attention with NLI-softmax variants.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`NLISoftmaxFunction(*args, **kwargs)`
:   Fast, quantization-friendly softmax: standard forward, exact softmax backward
    with numerical clamping for fp16/bf16 stability.

    ### Ancestors (in MRO)

    * torch.autograd.function.Function
    * torch.autograd.function._SingleLevelFunction
    * torch._C._FunctionBase
    * torch.autograd.function.FunctionCtx
    * torch.autograd.function._HookMixin

    ### Static methods

    `backward(ctx: Any, grad_output: torch.Tensor) ‑> tuple[torch.Tensor | None, None]`
    :   Define a formula for differentiating the operation with backward mode automatic differentiation.
        
        This function is to be overridden by all subclasses.
        (Defining this function is equivalent to defining the ``vjp`` function.)
        
        It must accept a context :attr:`ctx` as the first argument, followed by
        as many outputs as the :func:`forward` returned (None will be passed in
        for non tensor outputs of the forward function),
        and it should return as many tensors, as there were inputs to
        :func:`forward`. Each argument is the gradient w.r.t the given output,
        and each returned value should be the gradient w.r.t. the
        corresponding input. If an input is not a Tensor or is a Tensor not
        requiring grads, you can just pass None as a gradient for that input.
        
        The context can be used to retrieve tensors saved during the forward
        pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
        of booleans representing whether each input needs gradient. E.g.,
        :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
        first input to :func:`forward` needs gradient computed w.r.t. the
        output.

    `forward(ctx: Any, x: torch.Tensor, dim: int = -1) ‑> torch.Tensor`
    :   Define the forward of the custom autograd Function.
        
        This function is to be overridden by all subclasses.
        There are two ways to define forward:
        
        Usage 1 (Combined forward and ctx)::
        
            @staticmethod
            def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
                pass
        
        - It must accept a context ctx as the first argument, followed by any
          number of arguments (tensors or other types).
        - See :ref:`combining-forward-context` for more details
        
        Usage 2 (Separate forward and ctx)::
        
            @staticmethod
            def forward(*args: Any, **kwargs: Any) -> Any:
                pass
        
        
            @staticmethod
            def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
                pass
        
        - The forward no longer accepts a ctx argument.
        - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
          staticmethod to handle setting up the ``ctx`` object.
          ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
          to the forward.
        - See :ref:`extending-autograd` for more details
        
        The context can be used to store arbitrary data that can be then
        retrieved during the backward pass. Tensors should not be stored
        directly on `ctx` (though this is not currently enforced for
        backward compatibility). Instead, tensors should be saved either with
        :func:`ctx.save_for_backward` if they are intended to be used in
        ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
        if they are intended to be used for in ``jvp``.

`NMFCompressor(model: nn.Module | None = None, rank: int = 64, max_iter: int = 100, lora_r: int = 0, lora_alpha: float = 1.0, cfg: Any | None = None, target_layers: list[str] | None = None, target_attention_only: bool = False, max_error: float = 0.7, fallback_to_original: bool = True, **kwargs: Any)`
:   NMF Compressor: decomposes linear layer magnitudes into non-negative
    low-rank factors, preserving the sign pattern of the original weight.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress_linear(self, linear: nn.Module) ‑> neuralops.compression_engine.nmf_compressor.NMFLinear`
    :

`NMFLinear(in_features: int, out_features: int, rank: int = 64, bias: bool = True, dtype: torch.dtype | None = None, device: torch.device | None = None, lora_r: int = 0, lora_alpha: float = 1.0)`
:   Linear layer with weight factorized via sign-aware NMF:
    
        W = H @ V * W_sign
    
    where H ≥ 0, V ≥ 0, and W_sign ∈ {-1, +1} stores the sign of the original
    weight. Since `torch.sign` cannot be reconstructed by NMF, we factor the
    magnitude |W| with non-negative factors and remember the sign separately.
    
    Storage: H (out × r), V (r × in), W_sign (out × in as int8 or fp16).
    Note: W_sign storage is non-trivial (out × in elements). If storage is
    the priority, use a signed-NMF variant that avoids storing W_sign.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_linear(linear: nn.Module, rank: int = 64, max_iter: int = 30, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.nmf_compressor.NMFLinear`
    :   Fit sign-aware NMF on the given Linear / Conv1D / LoRALinear module
        using multiplicative updates (Lee & Seung 2001) with NNDSVD init.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix: H @ V * W_sign.

    ### Methods

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

`OATSCompressor(model: nn.Module, tokenizer: Any | None = None, sparsity: float = 0.5, rank_ratio: float = 0.2, num_iters: int = 4, device: str | torch.device | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, structured_24: bool = True, max_error: float = 0.85, fallback_to_original: bool = True, skip_attention_qkv: bool = False, target_attention_only: bool = False, target_modules: list[str] | None = None, lora_r: int = 0, lora_alpha: float = 1.0, min_rank: int = 4, **kwargs: Any)`
:   OATS Compressor: Outlier-aware low-rank + structured sparse matrix decomposition.
    
    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error (default 0.85).
    - Safe fallback to original weights by default (fallback_to_original=True).
    - Robust Conv1D detection by class name and nf attribute without unreliable shape heuristics.
    - Unified critical-layer protection via should_compress_linear.
    - GQA-aware skipping to protect asymmetric attention heads.
    - Tied / shared weight protection tracking parameter IDs.
    - Trainable LoRA adapter integration with base weights frozen.
    - FP32 numerical stability with NaN/Inf guards.
    - Post-compression forward sanity check under torch.no_grad().
    - Comprehensive per-layer stats and error tracking.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Executes OATS decomposition across compressible Linear and Conv1D modules.

`OATSLoRALinear(in_features: int, out_features: int, weight: torch.Tensor, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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
    :   Linear layer containing OATS decomposed base weights (frozen)
        with optional trainable LoRA adapters (lora_A, lora_B).

    ### Static methods

    `from_linear(linear: nn.Module, weight: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, **kwargs: Any) ‑> neuralops.compression_engine.oats_compressor.OATSLoRALinear`
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

`OptimalSubspaceLoRALinear(in_features: int, out_features: int, weight: torch.Tensor, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Frozen optimal-subspace-projected base weight + optional trainable LoRA.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_linear(linear: nn.Module, weight: torch.Tensor, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.optimal_subspace_projection.OptimalSubspaceLoRALinear`
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

`OptimalSubspaceProjectionCompressor(model: nn.Module, calibration_loader: Any | None = None, rank_ratio: float = 0.5, power_iters: int = 2, tokenizer: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, max_error: float = 0.5, fallback_to_original: bool = True, target_attention_only: bool = False, target_modules: list[str] | None = None, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = True, min_rank: int = 4, **kwargs: Any)`
:   Optimal Subspace Projection Compressor via activation-aware randomized SVD.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`Path(*args)`
:   PurePath subclass that can make system calls.
    
    Path represents a filesystem path but unlike PurePath, also offers
    methods to do system calls on path objects. Depending on your system,
    instantiating a Path will return either a PosixPath or a WindowsPath
    object. You can also instantiate a PosixPath or WindowsPath directly,
    but cannot instantiate a WindowsPath on a POSIX system or vice versa.

    ### Ancestors (in MRO)

    * pathlib.PurePath

    ### Descendants

    * pathlib.PosixPath
    * pathlib.WindowsPath

    ### Static methods

    `cwd()`
    :   Return a new path pointing to the current working directory.

    `from_uri(uri)`
    :   Return a new path from the given 'file' URI.

    `home()`
    :   Return a new path pointing to expanduser('~').

    ### Instance variables

    `info`
    :   A PathInfo object that exposes the file type and other file attributes
        of this path.

    ### Methods

    `absolute(self)`
    :   Return an absolute version of this path
        No normalization or symlink resolution is performed.
        
        Use resolve() to resolve symlinks and remove '..' segments.

    `as_uri(self)`
    :   Return the path as a URI.

    `chmod(self, mode, *, follow_symlinks=True)`
    :   Change the permissions of the path, like os.chmod().

    `copy(self, target, **kwargs)`
    :   Recursively copy this file or directory tree to the given destination.

    `copy_into(self, target_dir, **kwargs)`
    :   Copy this file or directory tree into the given existing directory.

    `exists(self, *, follow_symlinks=True)`
    :   Whether this path exists.
        
        This method normally follows symlinks; to check whether a symlink exists,
        add the argument follow_symlinks=False.

    `expanduser(self)`
    :   Return a new path with expanded ~ and ~user constructs
        (as returned by os.path.expanduser)

    `glob(self, pattern, *, case_sensitive=None, recurse_symlinks=False)`
    :   Iterate over this subtree and yield all existing files (of any
        kind, including directories) matching the given relative pattern.

    `group(self, *, follow_symlinks=True)`
    :   Return the group name of the file gid.

    `hardlink_to(self, target)`
    :   Make this path a hard link pointing to the same file as *target*.
        
        Note the order of arguments (self, target) is the reverse of os.link's.

    `is_block_device(self)`
    :   Whether this path is a block device.

    `is_char_device(self)`
    :   Whether this path is a character device.

    `is_dir(self, *, follow_symlinks=True)`
    :   Whether this path is a directory.

    `is_fifo(self)`
    :   Whether this path is a FIFO.

    `is_file(self, *, follow_symlinks=True)`
    :   Whether this path is a regular file (also True for symlinks pointing
        to regular files).

    `is_junction(self)`
    :   Whether this path is a junction.

    `is_mount(self)`
    :   Check if this path is a mount point

    `is_socket(self)`
    :   Whether this path is a socket.

    `is_symlink(self)`
    :   Whether this path is a symbolic link.

    `iterdir(self)`
    :   Yield path objects of the directory contents.
        
        The children are yielded in arbitrary order, and the
        special entries '.' and '..' are not included.

    `lchmod(self, mode)`
    :   Like chmod(), except if the path points to a symlink, the symlink's
        permissions are changed, rather than its target's.

    `lstat(self)`
    :   Like stat(), except if the path points to a symlink, the symlink's
        status information is returned, rather than its target's.

    `mkdir(self, mode=511, parents=False, exist_ok=False)`
    :   Create a new directory at this given path.

    `move(self, target)`
    :   Recursively move this file or directory tree to the given destination.

    `move_into(self, target_dir)`
    :   Move this file or directory tree into the given existing directory.

    `open(self, mode='r', buffering=-1, encoding=None, errors=None, newline=None)`
    :   Open the file pointed to by this path and return a file object, as
        the built-in open() function does.

    `owner(self, *, follow_symlinks=True)`
    :   Return the login name of the file owner.

    `read_bytes(self)`
    :   Open the file in bytes mode, read it, and close the file.

    `read_text(self, encoding=None, errors=None, newline=None)`
    :   Open the file in text mode, read it, and close the file.

    `readlink(self)`
    :   Return the path to which the symbolic link points.

    `rename(self, target)`
    :   Rename this path to the target path.
        
        The target path may be absolute or relative. Relative paths are
        interpreted relative to the current working directory, *not* the
        directory of the Path object.
        
        Returns the new Path instance pointing to the target path.

    `replace(self, target)`
    :   Rename this path to the target path, overwriting if that path exists.
        
        The target path may be absolute or relative. Relative paths are
        interpreted relative to the current working directory, *not* the
        directory of the Path object.
        
        Returns the new Path instance pointing to the target path.

    `resolve(self, strict=False)`
    :   Make the path absolute, resolving all symlinks on the way and also
        normalizing it.

    `rglob(self, pattern, *, case_sensitive=None, recurse_symlinks=False)`
    :   Recursively yield all existing files (of any kind, including
        directories) matching the given relative pattern, anywhere in
        this subtree.

    `rmdir(self)`
    :   Remove this directory.  The directory must be empty.

    `samefile(self, other_path)`
    :   Return whether other_path is the same or not as this file
        (as returned by os.path.samefile()).

    `stat(self, *, follow_symlinks=True)`
    :   Return the result of the stat() system call on this path, like
        os.stat() does.

    `symlink_to(self, target, target_is_directory=False)`
    :   Make this path a symlink pointing to the target path.
        Note the order of arguments (link, target) is the reverse of os.symlink.

    `touch(self, mode=438, exist_ok=True)`
    :   Create this file with the given access mode, if it doesn't exist.

    `unlink(self, missing_ok=False)`
    :   Remove this file or link.
        If the path is a directory, use rmdir() instead.

    `walk(self, top_down=True, on_error=None, follow_symlinks=False)`
    :   Walk the directory tree from this directory, similar to os.walk().

    `write_bytes(self, data)`
    :   Open the file in bytes mode, write to it, and close the file.

    `write_text(self, data, encoding=None, errors=None, newline=None)`
    :   Open the file in text mode, write to it, and close the file.

`ProxSparsePrunedLoRALinear(in_features: int, out_features: int, pruned_weight: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Frozen pruned base weight + trainable LoRA adapters.
    
    NOTE: `weight` is the ALREADY-PRUNED weight matrix. Forward does not
    re-apply the mask. `mask` is kept only as a compact bool buffer for
    downstream analysis / merge operations.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, pruned_weight: torch.Tensor, mask: torch.Tensor, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.proxsparse.ProxSparsePrunedLoRALinear`
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

`ProxSparseCompressor(model: nn.Module, sparsity: float = 0.5, lambda_prox: float = 0.01, steps: int = 5, learning_rate: float = 0.5, structured_24: bool = False, block_size: tuple[int, int] = (1, 1), skip_attention_qkv: bool = True, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, max_error: float | None = None, fallback_to_original: bool = True, min_numel: int = 16, target_modules: list[str] | None = None, target_attention_only: bool = False, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   ProxSparse: proximal gradient sparsity with structured / 2:4 / percentile
    constraint, optional LoRA fine-tuning, and safe fallback.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Return (compressed_model, statistics).

`ProxSparsePruner(model: nn.Module, sparsity: float = 0.5, lambda_prox: float = 0.01, steps: int = 5, learning_rate: float = 0.5, structured_24: bool = False, block_size: tuple[int, int] = (1, 1), skip_attention_qkv: bool = True, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, max_error: float | None = None, fallback_to_original: bool = True, min_numel: int = 16, target_modules: list[str] | None = None, target_attention_only: bool = False, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   ProxSparse: proximal gradient sparsity with structured / 2:4 / percentile
    constraint, optional LoRA fine-tuning, and safe fallback.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

`PrunedChannelLinear(in_features: int, original_out_features: int, kept_idx: torch.Tensor, W_kept: torch.Tensor, bias_kept: torch.Tensor | None = None)`
:   Linear layer with pruned output channels.
    
    Stores only the kept rows of the effective weight matrix:
      - W_kept:      (k, in_features)  fp16/fp32   -- only kept rows
      - bias_kept:   (k,)              optional
      - kept_idx:    (k,)              int64  -- output positions of kept rows
      - original_out_features: int
    
    Forward:
      h = F.linear(x, W_kept, bias_kept)          # (..., k)
      out = zeros(..., original_out_features)
      out.scatter_(-1, kept_idx, h)               # (..., original_out)
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention`
    :   The type of the None singleton.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstruct effective (original_out, in) weight with zeros in pruned rows.

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

`ROASTCompressor(model: nn.Module, n_buckets: int = 1024, tile_size: int = 32, skip_attention_qkv: bool = True, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, target_modules: list[str] | None = None, target_attention_only: bool = False, max_error: float = 0.5, fallback_to_original: bool = True, min_numel: int = 16, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Replace eligible linear and Conv1D layers with ROASTLinear (hardware-aware weight sharing).

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress the target modules using ROAST tiling with LoRA support.

`ROASTLinear(in_features: int, out_features: int, n_buckets: int = 1024, tile_size: int = 32, bias: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Linear layer with weight sharing based on tiles (blocks) rather than
    individual scalar weights. Each tile of size (tile_size, tile_size) shares
    a single weight from a small table, with optional trainable LoRA adapters.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, n_buckets: int = 1024, tile_size: int = 32, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.roast_hashed_linear.ROASTLinear`
    :   Initialize from a pretrained nn.Linear, Conv1D, or LoRALinear base_layer.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix (out_features, in_features) with evaluation caching.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Vectorized forward pass without per-element python loops.

`RPCACompressor(model: nn.Module, rank: int = 64, mu: float = 0.001, max_iter: int = 5, min_rank: int = 8, skip_attention_qkv: bool = False, target_attention_only: bool = False, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, target_modules: list[str] | None = None, max_error: float = 0.5, fallback_to_original: bool = True, min_numel: int = 16, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   Robust Principal Component Analysis (RPCA) Low-Rank Compressor.
        
        Decomposes weight matrices W into low-rank component L and sparse outlier matrix S
        via Principal Component Pursuit / alternating soft-thresholding:
            min ||L||_* + lambda ||S||_1  s.t.  W = L + S
        Factorizes the low-rank component L into symmetric factors (U * sqrt(Sigma), sqrt(Sigma) * V^T)
        and replaces nn.Linear or Conv1D with factorized SVDLinear modules with trainable LoRA fine-tuning adapters.

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress all linear layers in the model using RPCA decomposition.

    `compress_linear(self, layer: nn.Module, rank: int) ‑> neuralops.compression_engine.aa_svd_compressor.SVDLinear | None`
    :   Decompose a single linear layer via RPCA and return factorized SVDLinear with LoRA.

`ReplaceMeCompressor(model: nn.Module, remove_layers: int = 2, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, fallback_to_original: bool = True, device: str | torch.device | None = None, config: Any | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Replaces the trailing N layers of a Transformer with a lightweight
    LayerNorm + Linear residual block.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

`ReplacementBlock(hidden_size: int, lora_r: int = 0, lora_alpha: float = 1.0, return_tuple: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Drop-in replacement for a transformer layer.
    
    Preserves tensor shapes and maintains signal propagation via a residual
    connection with optional LoRA fine-tuning adapters.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, hidden_states: torch.Tensor | tuple[torch.Tensor, ...] | Any, *args: Any, **kwargs: Any) ‑> torch.Tensor | tuple[torch.Tensor, ...]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`ResSVDCompressor(model: nn.Module, base_rank: int = 32, residual_rank: int = 8, scaling: float = 1.0, factorize_base: bool = True, freeze_base: bool = True, target_modules: list[str] | None = None, **kwargs: Any)`
:   Spectral Residual LoRA (Res-SVD) Compressor.
    
    Replaces standard Linear layers with ResSVDLinear modules, truncating base weights
    at rank r (frozen) and initializing adapter matrices A, B directly from the residual error tail.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Descendants

    * neuralops.compression_engine.res_svd.ResSVDCompressorRegistered
    * neuralops.compression_engine.res_svd.SpectralResidualLoRA

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress target linear layers into ResSVDLinear modules.
        
        Returns:
            Tuple[nn.Module, Dict[str, Any]]: Compressed model and compression statistics.

`ResSVDCompressorRegistered(model: nn.Module, base_rank: int = 32, residual_rank: int = 8, scaling: float = 1.0, factorize_base: bool = True, freeze_base: bool = True, target_modules: list[str] | None = None, **kwargs: Any)`
:   Spectral Residual LoRA (Res-SVD) Compressor.
    
    Replaces standard Linear layers with ResSVDLinear modules, truncating base weights
    at rank r (frozen) and initializing adapter matrices A, B directly from the residual error tail.

    ### Ancestors (in MRO)

    * neuralops.compression_engine.res_svd.ResSVDCompressor
    * neuralops.core.base.BaseCompressor

`ResSVDLinear(in_features: int, out_features: int, base_rank: int = 32, residual_rank: int = 8, scaling: float = 1.0, factorize_base: bool = True, bias: bool = True, dtype: torch.dtype = torch.float32, device: torch.device | None = None)`
:   Spectral Residual LoRA (Res-SVD) Linear Layer.
    
    Executes forward pass as:
    y = x @ W_base^T + scaling * (x @ A^T) @ B^T + bias
    where W_base is either factorized as (base_V, base_U) with requires_grad=False,
    and (lora_A, lora_B) are trainable adapter matrices initialized from the residual tail.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Linear, base_rank: int = 32, residual_rank: int = 8, scaling: float = 1.0, factorize_base: bool = True, freeze_base: bool = True) ‑> neuralops.compression_engine.res_svd.ResSVDLinear`
    :   Construct ResSVDLinear by decomposing a pretrained nn.Linear weight:
        1. Base factors U_r, V_r from the dominant top-r singular components.
        2. Residual adapter factors A, B directly initialized from the residual tail (r+1 to r+k).
        
        Args:
            linear: Source dense linear layer.
            base_rank: Rank r for the frozen base matrix.
            residual_rank: Rank k for the trainable residual adapter.
            scaling: Multiplier alpha / k for adapter output.
            factorize_base: Whether to store base weights as factorized U, V matrices.
            freeze_base: Whether to freeze base factors (requires_grad=False).
        
        Returns:
            ResSVDLinear: Spectral residual LoRA module.

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
    :   Forward execution:
        y = Base(x) + scaling * (x @ A^T) @ B^T + bias

    `get_residual_error_norm(self, original_weight: torch.Tensor | None = None) ‑> float`
    :   Calculate relative Frobenius norm reconstruction error:
        || W_orig - (W_base + scaling * B @ A) ||_F / || W_orig ||_F

    `merge_weights(self) ‑> torch.nn.modules.linear.Linear`
    :   Merge frozen base weights and trainable residual adapter weights into a dense nn.Linear.
        
        Returns:
            nn.Linear: Standard dense linear layer with merged weights.

    `reset_parameters(self) ‑> None`
    :   Kaiming uniform parameter initialization for base and zero for adapters.

`RotationalSliceGPTCompressor(model: nn.Module | None = None, r_ratio: float = 0.4, lambda_reg: float = 0.01, slice_ratio: float | None = None, quant_bits: int = 4, max_error: float = 0.2, fallback_to_original: bool = True, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, target_modules: list[str] | None = None, target_attention_only: bool = False, seed: int = 42, **kwargs: Any)`
:   Rotational-SliceGPT-INT4 hybrid compression engine.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress(self, *args: Any, **kwargs: Any) ‑> Any`
    :   Return (compressed_model, statistics).

    `compress_linear(self, linear_layer: nn.Linear, X: torch.Tensor | None = None) ‑> neuralops.compression_engine.rotational_slicegpt_int4.RotationalSliceGPTINT4Linear`
    :   Convenience wrapper for compressing a single nn.Linear.

    `compress_matrix(self, W: torch.Tensor, X: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]`
    :   Returns:
            W_quant_indices:  (D_out, r) int8
            W_quant_centroids:(C,)       fp32
            Q_P_r:            (D_in, r)  fp32
            rel_error:        float

    `compress_tensor(self, W: torch.Tensor, X: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]`
    :   Returns:
            W_quant_indices:  (D_out, r) int8
            W_quant_centroids:(C,)       fp32
            Q_P_r:            (D_in, r)  fp32
            rel_error:        float

    `get_orthogonal_matrix(self, d_in: int, device: torch.device) ‑> torch.Tensor`
    :   Deterministic QR-based orthogonal Q (RS-FIX-05).

`RotationalSliceGPTINT4Compressor(model: nn.Module | None = None, r_ratio: float = 0.4, lambda_reg: float = 0.01, slice_ratio: float | None = None, quant_bits: int = 4, max_error: float = 0.2, fallback_to_original: bool = True, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, target_modules: list[str] | None = None, target_attention_only: bool = False, seed: int = 42, **kwargs: Any)`
:   Rotational-SliceGPT-INT4 hybrid compression engine.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress(self, *args: Any, **kwargs: Any) ‑> Any`
    :   Return (compressed_model, statistics).

    `compress_linear(self, linear_layer: nn.Linear, X: torch.Tensor | None = None) ‑> neuralops.compression_engine.rotational_slicegpt_int4.RotationalSliceGPTINT4Linear`
    :   Convenience wrapper for compressing a single nn.Linear.

    `compress_matrix(self, W: torch.Tensor, X: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]`
    :   Returns:
            W_quant_indices:  (D_out, r) int8
            W_quant_centroids:(C,)       fp32
            Q_P_r:            (D_in, r)  fp32
            rel_error:        float

    `compress_tensor(self, W: torch.Tensor, X: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]`
    :   Returns:
            W_quant_indices:  (D_out, r) int8
            W_quant_centroids:(C,)       fp32
            Q_P_r:            (D_in, r)  fp32
            rel_error:        float

    `get_orthogonal_matrix(self, d_in: int, device: torch.device) ‑> torch.Tensor`
    :   Deterministic QR-based orthogonal Q (RS-FIX-05).

`RotationalSliceGPTInt4Compressor(model: nn.Module | None = None, r_ratio: float = 0.4, lambda_reg: float = 0.01, slice_ratio: float | None = None, quant_bits: int = 4, max_error: float = 0.2, fallback_to_original: bool = True, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, target_modules: list[str] | None = None, target_attention_only: bool = False, seed: int = 42, **kwargs: Any)`
:   Rotational-SliceGPT-INT4 hybrid compression engine.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress_linear(self, linear_layer: nn.Linear, X: torch.Tensor | None = None) ‑> neuralops.compression_engine.rotational_slicegpt_int4.RotationalSliceGPTINT4Linear`
    :   Convenience wrapper for compressing a single nn.Linear.

    `compress_matrix(self, W: torch.Tensor, X: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]`
    :   Returns:
            W_quant_indices:  (D_out, r) int8
            W_quant_centroids:(C,)       fp32
            Q_P_r:            (D_in, r)  fp32
            rel_error:        float

    `compress_tensor(self, W: torch.Tensor, X: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]`
    :   Returns:
            W_quant_indices:  (D_out, r) int8
            W_quant_centroids:(C,)       fp32
            Q_P_r:            (D_in, r)  fp32
            rel_error:        float

    `get_orthogonal_matrix(self, d_in: int, device: torch.device) ‑> torch.Tensor`
    :   Deterministic QR-based orthogonal Q (RS-FIX-05).

`RotationalSliceGPTINT4Linear(W_indices: torch.Tensor, W_centroids: torch.Tensor, Q_P_r: torch.Tensor, bias: torch.Tensor | None = None)`
:   Linear layer storing:
      - Q_P_r:        (D_in, r) fp16/32 — the projection matrix
      - W_indices:    (D_out, r) int8   — quantized weight in sliced space
      - W_centroids:  (C,)       fp32   — per-tensor centroid table
      - bias:         (D_out,)   or None
    
    Forward: y = (x @ Q_P_r) → dequantize(W_indices) → @ .T → + bias
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `weight: torch.Tensor`
    :   Effective weight matrix (out_features, in_features).

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`SAESCompressor(model: nn.Module, rank: int | None = None, rank_ratio: float = 0.5, correcting_steps: int = 2, target_modules: list[str] | None = None, target_attention_only: bool = False, max_error: float = 0.6, fallback_to_original: bool = True, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   SAES-SVD v2 Compressor with iterative spectrum-aware residual correction,
        Frobenius validation, and safe fallback.

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress all linear and Conv1D layers in the model.

    `compress_linear(self, layer: nn.Module, rank: int | None = None, name: str = '') ‑> neuralops.compression_engine.aa_svd_compressor.SVDLinear | None`
    :   Decompose a single linear or Conv1D layer with spectrum-aware error suppression.

`SAESSVDCompressorV2(model: nn.Module, rank: int | None = None, rank_ratio: float = 0.5, correcting_steps: int = 2, target_modules: list[str] | None = None, target_attention_only: bool = False, max_error: float = 0.6, fallback_to_original: bool = True, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   SAES-SVD v2 Compressor with iterative spectrum-aware residual correction,
        Frobenius validation, and safe fallback.

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress all linear and Conv1D layers in the model.

    `compress_linear(self, layer: nn.Module, rank: int | None = None, name: str = '') ‑> neuralops.compression_engine.aa_svd_compressor.SVDLinear | None`
    :   Decompose a single linear or Conv1D layer with spectrum-aware error suppression.

`SALRCompressor(model: nn.Module, rank: int = 64, sparsity: float = 0.5, min_rank: int = 8, skip_attention_qkv: bool = False, target_attention_only: bool = False, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, target_modules: list[str] | None = None, max_error: float = 0.5, fallback_to_original: bool = True, min_numel: int = 16, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   SALR Compressor: Structure-Aware Low-Rank and Sparse Pruning.
        
        Factorizes dense nn.Linear or Conv1D layers into symmetric low-rank factors
        (U * sqrt(Sigma), sqrt(Sigma) * V^T) and applies structured sparse pruning
        to trailing singular components in the bottleneck with LoRA adapters for recovery.

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress all linear layers in the model.

    `compress_linear(self, layer: nn.Module, rank: int) ‑> neuralops.compression_engine.salr.SALRLayer | None`
    :   Decompose a single linear or Conv1D layer into SALRLayer with validation.

`SALRLayer(in_features: int, out_features: int, rank: int, sparsity: float = 0.5, bias: bool = True, dtype: torch.dtype = torch.float32, device: torch.device | None = None, lora_r: int = 0, lora_alpha: float = 1.0)`
:   Structure-Aware Low-Rank + Sparse Layer (SALR).
    
    Combines truncated SVD rank reduction with sparse singular component selection
    and optional trainable LoRA fine-tuning adapters:
        y = (x @ (V * mask).T) @ U.T + bias + (lora_B @ lora_A @ x) * scaling
    where U: [out_features, rank], V: [rank, in_features], mask: [rank, 1].
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Instance variables

    `weight: torch.Tensor`
    :   Effective reconstructed weight matrix.

    ### Methods

    `extra_repr(self) ‑> str`
    :   Return the extra representation of the module.
        
        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Two-stage forward pass with sparse bottleneck mask and optional LoRA branch.

    `reset_parameters(self) ‑> None`
    :

    `set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) ‑> None`
    :

`SDRMCompressor(model: nn.Module, total_param_ratio: float = 0.7, min_rank: int = 8, skip_attention_qkv: bool = False, target_attention_only: bool = False, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, target_modules: list[str] | None = None, max_error: float = 0.5, fallback_to_original: bool = True, min_numel: int = 16, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   SDRM: Scaled Dimension Rank Modulation.
        Assigns layer rank proportional to parameter budget for edge inference and factorizes
        target layers into low-rank SVDLinear modules with optional LoRA fine-tuning adapters.

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress linear layers using budget-proportional rank assignment.

    `compress_linear(self, layer: nn.Module, rank: int) ‑> neuralops.compression_engine.aa_svd_compressor.SVDLinear | None`
    :   Decompose a single linear or Conv1D layer into SVDLinear in FP32 with validation.

`SLiMCompressor(model: nn.Module, bits: int = 4, sparsity: float = 0.5, rank: int = 8, structured_24: bool = True, target_attention_only: bool = False, max_error: float = 0.5, fallback_to_original: bool = True, target_modules: list[str] | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   SLiM Compressor: One-shot joint b-bit quantization, 2:4 structured sparsity, and low-rank residual recovery.

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Executes SLiM compression pipeline across all Linear and Conv1D layers.

`STEFakeQuantize(*args, **kwargs)`
:   Straight-Through Estimator (STE) fake-quantization autograd function.
    Clamps inputs to [qmin, qmax] in discrete integer space, de-quantizes back,
    and passes gradients through uninhibited across all channels.

    ### Ancestors (in MRO)

    * torch.autograd.function.Function
    * torch.autograd.function._SingleLevelFunction
    * torch._C._FunctionBase
    * torch.autograd.function.FunctionCtx
    * torch.autograd.function._HookMixin

    ### Static methods

    `backward(ctx: Any, grad_output: torch.Tensor) ‑> tuple[torch.Tensor | None, ...]`
    :   Define a formula for differentiating the operation with backward mode automatic differentiation.
        
        This function is to be overridden by all subclasses.
        (Defining this function is equivalent to defining the ``vjp`` function.)
        
        It must accept a context :attr:`ctx` as the first argument, followed by
        as many outputs as the :func:`forward` returned (None will be passed in
        for non tensor outputs of the forward function),
        and it should return as many tensors, as there were inputs to
        :func:`forward`. Each argument is the gradient w.r.t the given output,
        and each returned value should be the gradient w.r.t. the
        corresponding input. If an input is not a Tensor or is a Tensor not
        requiring grads, you can just pass None as a gradient for that input.
        
        The context can be used to retrieve tensors saved during the forward
        pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
        of booleans representing whether each input needs gradient. E.g.,
        :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
        first input to :func:`forward` needs gradient computed w.r.t. the
        output.

    `forward(ctx: Any, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, qmin: float | torch.Tensor, qmax: float | torch.Tensor) ‑> torch.Tensor`
    :   Define the forward of the custom autograd Function.
        
        This function is to be overridden by all subclasses.
        There are two ways to define forward:
        
        Usage 1 (Combined forward and ctx)::
        
            @staticmethod
            def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
                pass
        
        - It must accept a context ctx as the first argument, followed by any
          number of arguments (tensors or other types).
        - See :ref:`combining-forward-context` for more details
        
        Usage 2 (Separate forward and ctx)::
        
            @staticmethod
            def forward(*args: Any, **kwargs: Any) -> Any:
                pass
        
        
            @staticmethod
            def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
                pass
        
        - The forward no longer accepts a ctx argument.
        - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
          staticmethod to handle setting up the ``ctx`` object.
          ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
          to the forward.
        - See :ref:`extending-autograd` for more details
        
        The context can be used to store arbitrary data that can be then
        retrieved during the backward pass. Tensors should not be stored
        directly on `ctx` (though this is not currently enforced for
        backward compatibility). Instead, tensors should be saved either with
        :func:`ctx.save_for_backward` if they are intended to be used in
        ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
        if they are intended to be used for in ``jvp``.

`SVDLLMCompressor(model: nn.Module, rank_ratio: float = 0.2, rank: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, max_error: float = 0.5, fallback_to_original: bool = False, calibration_texts: list[str] | None = None, tokenizer: Any | None = None, device: str | torch.device | None = None, target_attention_only: bool = True, recovery_steps: int = 50, learning_rate: float = 0.0002, weight_decay: float = 0.01, batch_size: int = 8, **kwargs: Any)`
:   SVD-LLM (ICLR 2025) Compressor with Cholesky activation whitening,
    sequential LoRA recovery fine-tuning, and fixed rank parameter ratios.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress the target model using full SVD-LLM Cholesky whitening and structural replacement.

    `compute_rank_from_ratio(self, m: int, n: int, ratio: float) ‑> int`
    :   Compute low rank r from parameter retention ratio R:
        Original params = m * n
        Target params = R * m * n
        Factorized params = r * (m + n)
        r = (R * m * n) / (m + n)

`SVDLLMLinear(in_features: int, out_features: int, rank: int, bias: bool = True, dtype: torch.dtype = torch.float32, device: torch.device | None = None, lora_rank: int = 16, lora_alpha: float = 32.0)`
:   SVD-LLM Factorized Linear Layer executing as:
    y = (x @ W_v.T + lora_v(x)) @ (W_u.T) + lora_u(x @ W_v.T) + bias
    
    Specifically, with sequential LoRA support:
    - W_u ∈ R^(d_out × r), W_v ∈ R^(r × d_in)
    - LoRA for W_u: A_u ∈ R^(r_L × r), B_u ∈ R^(d_out × r_L)
    - LoRA for W_v: A_v ∈ R^(r_L × d_in), B_v ∈ R^(r × r_L)
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `reset_base_parameters(self) ‑> None`
    :

    `reset_lora_parameters(self) ‑> None`
    :

    `set_active_lora_stage(self, stage: str | None) ‑> None`
    :   stage == 'stage1': train A_u, B_u only (W_v frozen)
        stage == 'stage2': train A_v, B_v only (W_u frozen)
        stage == 'both': train all LoRA adapters
        stage is None: freeze all LoRA adapters

    `set_factors(self, W_u: torch.Tensor, W_v: torch.Tensor, bias: torch.Tensor | None = None) ‑> None`
    :

`SVDLinear(in_features: int, out_features: int, rank: int, bias: bool = True, use_fused_kernel: bool = False, dtype: torch.dtype = torch.float32, device: torch.device | None = None, lora_rank: int = 0, lora_alpha: float = 1.0, lora_r: int | None = None, freeze_base: bool = False)`
:   Low-rank factorized linear layer:
        y = (x @ V.T) @ U.T + LoRA(x) * scaling + b
    Parameter count: r·(d_in + d_out) + LoRA params.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `use_lora: bool`
    :

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

    `freeze_base_weights(self) ‑> None`
    :

    `get_weight(self) ‑> torch.Tensor`
    :

    `reset_parameters(self) ‑> None`
    :

    `set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) ‑> None`
    :

`AdaptiveSVDLinear(in_features: int, out_features: int, rank: int, bias: bool = True, use_fused_kernel: bool = False, dtype: torch.dtype = torch.float32, device: torch.device | None = None, lora_rank: int = 0, lora_alpha: float = 1.0, lora_r: int | None = None, freeze_base: bool = False)`
:   Low-rank factorized linear layer:
        y = (x @ V.T) @ U.T + LoRA(x) * scaling + b
    Parameter count: r·(d_in + d_out) + LoRA params.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `use_lora: bool`
    :

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

    `freeze_base_weights(self) ‑> None`
    :

    `get_weight(self) ‑> torch.Tensor`
    :

    `reset_parameters(self) ‑> None`
    :

    `set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) ‑> None`
    :

`SVDLinearCompressor(model: nn.Module, rank: int = 192, max_error: float = 0.2, fallback_to_original: bool = True, protect_critical: bool = True, target_attention_only: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = False, target_modules: list[str] | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Compresses model by replacing Linear/Conv1D layers with SVDLinearLayer.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress linear layers with SVD factorization.

`SVDLinearLayer(linear_layer: nn.Module, rank: int, use_fused_kernel: bool = False, use_rms_norm: bool = False, eps: float = 1e-06, lora_r: int = 0, lora_alpha: float = 1.0)`
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
    :   Replace a linear layer with factorized U @ V, keeping top-`rank` components.
        Optionally dispatches fused in-SRAM kernel to eliminate memory roundtrips.
        Includes optional trainable LoRA adapters (lora_A, lora_B).

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix for compatibility.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Forward pass with optional fused kernel acceleration and LoRA branch.

`SVDSingleHead(hidden_size: int, svd_rank: int = 20)`
:   DEPRECATED single-head SVD attention.
    
    Performing SVD on live activations is O(N^2 d), numerically fragile, and
    theoretically incorrect as an attention replacement. Use the weight-based
    HybridSVDCompressor instead.
    
    Kept strictly for backwards compatibility. First use emits a warning.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `forward(self, hidden_states: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

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

`SimpleEntropyCoder()`
:   Lightweight entropy coder using empirical symbol frequency estimation.

    ### Static methods

    `compute_compression_ratio(tensor: torch.Tensor, uncompressed_bits: int = 16) ‑> float`
    :   Estimate theoretical compression ratio relative to FP16/FP32 baseline.

    `compute_entropy(tensor: torch.Tensor) ‑> float`
    :   Compute empirical Shannon entropy in bits per symbol.

`SliceGPTCompressor(model: nn.Module, tokenizer: Any | None = None, sparsity: float = 0.6, non_uniform: bool = False, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, target_attention_only: bool = False, device: str | torch.device | None = None, max_error: float = 0.5, fallback_to_original: bool = True, target_modules: list[str] | None = None, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, **kwargs: Any)`
:   SliceGPT Compression Strategy (real low-rank implementation).
    
    Performs orthogonal rotation of each target weight matrix using the
    eigendecomposition of its input activation covariance, then truncates
    to the top-r principal coordinates. The truncated weight is stored as a
    rank-r SVDLinear factorization, achieving genuine parameter reduction.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Applies SliceGPT rotation + rank-r truncation and replaces each weight
        with a real SVDLinear low-rank factorization.

`SliceGPTPCACompressor(model: nn.Module, calibration_loader: Any | None = None, sparsity: float = 0.3, tokenizer: Any | None = None, calibration_texts: list[str] | None = None, skip_attention_qkv: bool = False, target_attention_only: bool = False, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, max_error: float = 0.5, fallback_to_original: bool = True, min_numel: int = 16, target_modules: list[str] | None = None, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Production-hardened SliceGPT PCA-based compressor that performs activation-aware
    principal coordinate projection and dimension slicing on Linear and Conv1D layers.
    
    Replaces each sliced dense weight with a real SVDLinear low-rank factorization
    so that parameter count actually drops from d_out*d_in to (d_out+d_in)*keep.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compresses the model using PCA-based principal subspace projection and
        real low-rank factorization (SVDLinear).
        
        Returns (compressed_model, stats).

`SliceGPTPCALoRALinear(in_features: int, out_features: int, weight: torch.Tensor, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Linear layer containing sliced/PCA-projected base weights (frozen)
    with trainable LoRA adapters (lora_A, lora_B) for post-compression fine-tuning.
    Base weights are frozen with requires_grad=False; initial LoRA contribution is zero.
    
    NOTE: This wrapper stores the base weight densely. For real parameter
    reduction, the compressor now uses SVDLinear instead. This class is kept
    for backward compatibility only.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, sliced_weight: torch.Tensor, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.slicegpt_pca.SliceGPTPCALoRALinear`
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

`SparseGPTCompressor(model: nn.Module, tokenizer: Any | None = None, sparsity: float = 0.5, prunen: int = 2, prunem: int = 4, block_size: int = 128, percdamp: float = 0.01, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, target_attention_only: bool = False, device: str | torch.device | None = None, structured_24: bool = False, max_error: float = 0.5, fallback_to_original: bool = True, target_modules: list[str] | None = None, **kwargs: Any)`
:   In-memory PyTorch SparseGPT Compressor.
    
    Performs layer-by-layer second-order pruning with Optimal Brain Surgeon (OBS) weight
    compensation, supporting unstructured sparsity, block sparsity, and strict 2:4 structured sparsity.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Executes SparseGPT compression across all target linear layers.

`SpectralResidualLoRA(*args: Any, **kwargs: Any)`
:   PEFT-focused wrapper for Spectral Residual LoRA.

    ### Ancestors (in MRO)

    * neuralops.compression_engine.res_svd.ResSVDCompressor
    * neuralops.core.base.BaseCompressor

    ### Descendants

    * neuralops.compression_engine.res_svd.SpectralResidualLoRARegistered

`SpectralResidualLoRARegistered(*args: Any, **kwargs: Any)`
:   PEFT-focused wrapper for Spectral Residual LoRA.

    ### Ancestors (in MRO)

    * neuralops.compression_engine.res_svd.SpectralResidualLoRA
    * neuralops.compression_engine.res_svd.ResSVDCompressor
    * neuralops.core.base.BaseCompressor

`SubspaceLinear(in_features: int, out_features: int, rank: int = 64, bias: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Subspace Projection Factorized Linear Layer with optional LoRA adapter.
    
    W ≈ B @ A, where:
      A (rank, in_features):  orthonormal / projection basis
      B (out_features, rank): coefficient expansion matrix
    
    Forward:
      y = ((x @ A^T) @ B^T) + bias + (lora_B @ lora_A @ x) * scaling
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_linear(linear: nn.Module, rank: int = 64, method: str = 'svd', use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.subspace_projection.SubspaceLinear`
    :   Project linear or Conv1D weight matrix onto top-rank subspace.

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

    `get_weight(self) ‑> torch.Tensor`
    :

`SubspaceProjectionCompressor(model: nn.Module | None = None, rank: int = 64, method: str = 'svd', cfg: Any | None = None, target_layers: list[str] | None = None, target_attention_only: bool = False, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, max_error: float = 0.5, fallback_to_original: bool = True, **kwargs: Any)`
:   Subspace Projection Compression Engine.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress_linear(self, linear: nn.Module) ‑> neuralops.compression_engine.subspace_projection.SubspaceLinear`
    :

`SwiftSVDActivationCompressor(model: nn.Module, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, rank: int = 64, max_error: float = 0.5, fallback_to_original: bool = True, target_attention_only: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = False, min_rank: int = 4, target_modules: list[str] | None = None, num_calibration_batches: int = 8, max_calibration_tokens: int = 32768, use_fp64_cross_cov: bool = True, **kwargs: Any)`
:   Swift-SVD Activation-Aware Compressor.
    
    Captures empirical input-output cross-covariances across calibration data,
    derives activation-aligned subspaces, projects weights onto the principal
    activation subspace, and replaces dense nn.Linear / Conv1D layers with
    factorized SVDLinear modules.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress_linear(self, layer: nn.Module, act_pair: tuple[torch.Tensor, int] | None = None) ‑> neuralops.compression_engine.aa_svd_compressor.SVDLinear | None`
    :

`SwiftSVDCompressor(model: nn.Module, rank: int | None = None, rank_ratio: float = 0.5, target_modules: list[str] | None = None, max_error: float = 0.6, max_retries: int = 3, fallback_to_original: bool = True, validation_calibration: bool = False, calibration_threshold: float = 0.1, target_attention_only: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   Swift-SVD Compressor: Fast Closed-Form Truncated SVD.
        
        Decomposes dense weight matrices into rank-r factorized layers with symmetric
        square-root singular value splitting (U_r * sqrt(Sigma_r), sqrt(Sigma_r) * V_r^T)
        and replaces nn.Linear/Conv1D with factorized SVDLinear modules.

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress all eligible linear and Conv1D layers.

    `compress_linear(self, layer: nn.Module, rank: int | None = None, name: str = '') ‑> tuple[neuralops.compression_engine.aa_svd_compressor.SVDLinear | None, float, int]`
    :   Decompose a single linear or Conv1D layer via truncated SVD.
        Returns (svd_layer_or_None, relative_frobenius_error, final_rank).

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

`ToMoECompressor(model: nn.Module, num_experts: int = 4, top_k: int = 2, max_error: float = 0.2, lora_r: int = 0, lora_alpha: float = 16.0, freeze_experts: bool = True, target_attention_only: bool = False, target_modules: list[str] | None = None, **kwargs: Any)`
:   ToMoE Compressor: dynamically replaces dense feed-forward and projection layers
    with an MoE-fied equivalent.
    
    Args:
        model: PyTorch model.
        num_experts: Number of expert sub-layers to form (default: 4).
        top_k: Number of experts to route each token to (default: 2).
        max_error: Maximum allowed relative Frobenius error on reconstruction (default: 0.20).
        lora_r: LoRA adapter rank on experts and router (0 = disabled).
        lora_alpha: LoRA scaling factor (default: 16.0).
        freeze_experts: Whether to freeze expert base weights (default: True).
        target_attention_only: Whether to restrict to attention layers only.
        target_modules: Optional list of target layer name substrings.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `PROTECTED_KEYWORDS`
    :   The type of the None singleton.

    ### Methods

    `compress(self, orig_model: nn.Module | None = None) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress target layers into ToMoE Mixture of Experts layers with full error validation.

`ToMoELinear(original: nn.Module | None = None, in_features: int | None = None, out_features: int | None = None, weight: torch.Tensor | None = None, bias: torch.Tensor | None = None, num_experts: int = 4, top_k: int = 2, lora_r: int = 0, lora_alpha: float = 16.0, freeze_experts: bool = True, max_error: float = 0.2)`
:   A single MoE-fied linear layer: learned routing + expert sub-layers.
    
    Features:
    - Clusters output neurons using FP32 KMeans (with equal-division fallback).
    - Vectorized forward pass using gather and scatter_add_ for equal-sized experts,
      or vectorized index-dispatch for variable-sized experts.
    - Numerical stability in FP16, BF16, and FP32.
    - Exposes reconstructed effective weight and bias properties for validation.
    - Supports optional trainable LoRA adapters on experts and router.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Instance variables

    `bias: torch.Tensor | None`
    :   Reconstructed effective full bias vector (out_features,).

    `masks: list[torch.Tensor]`
    :

    `weight: torch.Tensor`
    :   Reconstructed effective full weight matrix (out_features, in_features).

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`UniQLCompressor(model: nn.Module, rank: int = 64, bits: int = 8, max_error: float = 0.2, max_retries: int = 3, fallback_to_original: bool = True, target_attention_only: bool = False, quantize_bias: bool = False, **kwargs: Any)`
:   UniQL: Joint Truncated SVD + Uniform Quantization compressor.
    
    Unlike the naive "copy back" variant, this compressor installs a real
    UniQLinear module that stores int8 U, S, Vt buffers. Parameter count
    and memory footprint decrease by roughly the SVD factor
    (out + in) * r / (out * in).

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`UniQLinear(in_features: int, out_features: int, rank: int, U_q: torch.Tensor, S_q: torch.Tensor, Vt_q: torch.Tensor, scale_u: torch.Tensor, scale_s: torch.Tensor, scale_v: torch.Tensor, bias: torch.Tensor | None = None, is_conv1d: bool = False, dtype: torch.dtype = torch.float32, device: torch.device | None = None)`
:   Replaces a Linear (or Conv1D) with a stored factorization:
    
        W ≈ U @ diag(S) @ Vt
        where U  : (out, r) int8, scale_u  (out,)
              S  : (r,)    int8, scale_s  scalar-per-rank
              Vt : (r, in) int8, scale_v  (in,)
    
    Forward reconstructs W_hat on the fly and applies F.linear / F.linear with
    transposed Conv1D weight. Numerically this is exactly the same
    reconstruction as the original UniQL, but now the storage is genuinely
    reduced from out*in fp16 to (out + in) * r int8 plus small scale vectors.
    
    Original bias (if any) is stored as-is unless quantize_bias=True.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed weight (out, in). Conv1D callers should transpose.

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