Module neuralops.compression_engine.armor_pruner
================================================
ARMOR: Adaptive Resilient Matrix Orthogonal Rank Pruning (2:4 Sparse Core + Low-Rank Correction).
Registered in COMPRESSOR_REGISTRY as 'armor_pruner', 'armor', and 'armor_compressor'.

Functions
---------

`armor_pruner_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Factory function for ARMOR pruner.

`compute_resilient_taylor_saliency(weight: torch.Tensor, grad: torch.Tensor | None = None, activations: torch.Tensor | None = None) ‑> torch.Tensor`
:   Resilient Taylor saliency with three-tier fallback:
      1. |W · g| if gradient is valid and nonzero.
      2. |W| · E[|x|] if activations available.
      3. |W| · (1 + σ(W)) as fallback.
    
    AR-FIX-12: `activations` may be a 1-D tensor of shape (in_features,) representing
    per-channel mean |x|. In that case, broadcast along the input dimension.

Classes
-------

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