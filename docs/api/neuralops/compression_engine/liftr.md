Module neuralops.compression_engine.liftr
=========================================
LIFTR – Linearized Function Trick Compressor.
Registered as 'liftr', 'liftr_compressor', 'liftr_class'.

Mathematical Formulation:
  Activation:    ReLU(x) → Identity(x), GELU(x) / SiLU(x) ≈ α·x
  Attention:     Softmax(QK^T/√d)V → (1/N) Σ V_i

Functions
---------

`liftr_compressor_entry(model: nn.Module, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Factory function for LIFTR compressor.

Classes
-------

`LIFTRCompressor(model: nn.Module, replace_attention: bool = False, gelu_coefficient: float = 0.8, validation_calibration: bool = False, calibration_threshold: float = 0.2, target_attention_only: bool = False, skip_should_compress_filter: bool = False, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `CRITICAL_MODULE_KEYWORDS`
    :   The type of the None singleton.

    `is_custom_attention: bool`
    :   The type of the None singleton.