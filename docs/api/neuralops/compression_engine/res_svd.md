Module neuralops.compression_engine.res_svd
===========================================
Spectral Residual Low-Rank Adaptation (Res-SVD / Spectral Residual LoRA).

Decomposes dense weight matrices into:
1. Frozen Base Low-Rank Factors: $W_{\text{base}} = U_r \Sigma_r V_r^\top$ (rank $r$, frozen).
2. Trainable Residual Adapters: Initialized directly from the top-$k$ singular directions of the
   residual error $E = W - W_{\text{base}} = \sum_{j=r+1}^{r+k} \sigma_j u_j v_j^\top$.
   $$A = \sqrt{\Sigma_{\text{tail}}} V_{\text{tail}}^\top, \quad B = U_{\text{tail}} \sqrt{\Sigma_{\text{tail}}}$$
   ensuring $\Delta W = B A \approx E$ at zero-shot initialization.

Functions
---------

`res_svd_compressor_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> torch.nn.modules.module.Module`
:   Res-SVD factory function.

`res_svd_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> torch.nn.modules.module.Module`
:   Res-SVD factory function.

`spectral_residual_lora_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> torch.nn.modules.module.Module`
:   Spectral Residual LoRA factory function.

Classes
-------

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

`SpectralResidualLoRA(*args: Any, **kwargs: Any)`
:   PEFT-focused wrapper for Spectral Residual LoRA.

    ### Ancestors (in MRO)

    * neuralops.compression_engine.res_svd.ResSVDCompressor
    * neuralops.core.base.BaseCompressor

    ### Descendants

    * neuralops.compression_engine.res_svd.SpectralResidualLoRARegistered