Module neuralops.compression_engine.erc_svd
===========================================
ERC-SVD: Error-Controlled SVD with Bounded Frobenius Truncation Error.
Registered as 'erc' and 'erc_svd'.

Bug-fix map:
  ER-FIX-01  fallback path handles svd_layer=None safely.
  ER-FIX-02  LoRA rank detected via lora_A.weight.shape[0] (PEFT-compatible).
  ER-FIX-03  forward sanity check error message corrected.
  ER-FIX-04  attention_mask included in forward check.
  ER-FIX-05  r_init clamped to max_r before storing.
  ER-FIX-06  erc_factory guarded, returns stats optionally.
  ER-FIX-07  no wasted SVDLayer when best_err > max_error and fallback enabled.
  ER-FIX-08  no silent mutation of non-target layers (only freeze when needed).

Functions
---------

`erc_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   ERC factory function.

Classes
-------

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