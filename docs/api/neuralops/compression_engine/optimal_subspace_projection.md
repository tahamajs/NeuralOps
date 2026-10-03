Module neuralops.compression_engine.optimal_subspace_projection
===============================================================
Optimal Subspace Projection: Activation-Aware Subspace Projection via Randomized SVD.

Registered in COMPRESSOR_REGISTRY as 'optimal_subspace_projection'.

Mathematical Formulation:
  1. Activation Covariance:      C = (1/N) X^T X + eps * I
  2. Cholesky Preconditioning:   C = L L^T,   M = W L
  3. Randomized Subspace Iter.:  Y = M Omega,  Y <- M (M^T Y),  Q = qr(Y)
  4. Low-Rank Projection:        B = Q^T M = U S V^T
     W_osp = (Q U_r S_r V_r^T) L^{-1}

Functions
---------

`inject_optimal_subspace_projection(model: Any = None, layer: Any = None, **kwargs: Any) ‑> Any`
:   

`optimal_subspace_projection_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   

Classes
-------

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

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Return (compressed_model, statistics).

`SubspaceProjectionCompressor(model: nn.Module, calibration_loader: Any | None = None, rank_ratio: float = 0.5, power_iters: int = 2, tokenizer: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, max_error: float = 0.5, fallback_to_original: bool = True, target_attention_only: bool = False, target_modules: list[str] | None = None, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = True, min_rank: int = 4, **kwargs: Any)`
:   Optimal Subspace Projection Compressor via activation-aware randomized SVD.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.