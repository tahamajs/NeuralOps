Module neuralops.compression_engine.mamba_shedder
=================================================
Mamba-Shedder: Sensitivity-Based Structured Pruning and Layer Shedding Compressor.

Registered as 'mamba_shedder', 'mamba_shedder_compressor', 'mamba_shedder_factory'.

Mathematical Formulation:
  1. Empirical Fisher:  I(X_j) = sqrt(mean_n X_{n,j}^2)
  2. Layer Residual:    Δ_l = ||X_{l+1} - X_l||_F / (||X_l||_F + ε)
  3. Channel Masking:   W_pruned = W ⊙ M_channel^T

⚠️  IMPORTANT  ⚠️

Channel masking alone does NOT reduce dense storage. To actually shrink
memory, this version replaces targeted Linear/Conv1D layers with a
`MambaShedderPrunedLinear` module that stores ONLY the retained channels
(via per-output-channel keep indices) plus LoRA adapters. The framework's
dense parameter counter will still read 1.0 for `compression_rate` because
the module signature is preserved; the true storage savings are reported in
`storage_compression_ratio` and `storage_reduction_pct`.

Functions
---------

`mamba_shedder_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   

Classes
-------

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