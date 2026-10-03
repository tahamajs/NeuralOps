Module neuralops.compression_engine.kvtc_compressor
===================================================
KVTC: KV Cache Transform Coding - PCA decorrelation + adaptive quantization
      + entropy coding for compact model weight / KV cache storage.

Paper: "KV Cache Transform Coding for Compact Storage in LLM Inference"
       Staniszewski & Lancucki, ICLR 2026.

NOTE ON SCOPE:
In this codebase, KVTC operates primarily as a static weight-matrix compressor
(PCA low-rank projection + uniform integer quantization + optional LoRA adaptation)
applied to linear layers at initialization time, rather than dynamic runtime KV cache
interception during autoregressive generation. For runtime KV cache transform coding,
the `KVTCStore` class is provided as a standalone utility.

Registered in COMPRESSOR_REGISTRY as:
  - 'kvtc' (primary)
  - 'kvtc_quantizer'
  - 'pca_quant'
  - 'pca_quant_compressor'

Functions
---------

`kvtc_compressor_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   KVTC compressor factory function.

`kvtc_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   KVTC compressor factory function.

`pca_quant_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   KVTC compressor factory function.

Classes
-------

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

`SimpleEntropyCoder()`
:   Lightweight entropy coder using empirical symbol frequency estimation.

    ### Static methods

    `compute_compression_ratio(tensor: torch.Tensor, uncompressed_bits: int = 16) ‑> float`
    :   Estimate theoretical compression ratio relative to FP16/FP32 baseline.

    `compute_entropy(tensor: torch.Tensor) ‑> float`
    :   Compute empirical Shannon entropy in bits per symbol.