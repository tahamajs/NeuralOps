Module neuralops.compression_engine.rotational_slicegpt_int4
============================================================
Rotational SliceGPT INT4 Compressor — Production-Hardened.

Combines orthogonal rotation, variance-based subspace slicing, and
second-order INT4 non-uniform quantization.

Mathematical formulation:
  1. Rotation:       W̃ = W Q,   X̃ = Qᵀ X,   Qᵀ Q = I
  2. Slicing:        P_r = [e_i] for top-r var(X̃) channels
                     W_sliced = W̃ P_r,   X_sliced = P_rᵀ X̃
  3. Hessian INT4:   H = 2 X_sliced X_slicedᵀ + λ I
                     W[:, j+1:r] -= (w_j - q(w_j)) · [H⁻¹]_{j, j+1:r} / [H⁻¹]_{jj}

Registered: 'rotational_slicegpt_int4', 'rotational_slicegpt',
            'rotational_slicegpt_int4_factory', 'rotational_slicegpt_factory'.

Functions
---------

`rotational_slicegpt_int4_factory(model: Any, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> Any`
:   

Classes
-------

`AstroQuantizer(bits: int = 4)`
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

`RotationalSliceGPTCompressor(model: nn.Module | None = None, r_ratio: float = 0.4, lambda_reg: float = 0.01, slice_ratio: float | None = None, quant_bits: int = 4, max_error: float = 0.2, fallback_to_original: bool = True, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, target_modules: list[str] | None = None, target_attention_only: bool = False, seed: int = 42, **kwargs: Any)`
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