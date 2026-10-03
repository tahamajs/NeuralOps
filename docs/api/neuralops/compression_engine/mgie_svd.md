Module neuralops.compression_engine.mgie_svd
============================================
MGIE-SVD: Maximum Information Entropy SVD Rank Selection.

Registered as 'mgie', 'mgie_svd', 'mgie_factory', 'mgie_svd_factory'.

Functions
---------

`mgie_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   MGIE factory function.

`mgie_svd_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   MGIE factory function.

Classes
-------

`MGIECompressor(model: nn.Module, rank: int | None = None, max_error: float = 0.5, fallback_to_original: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = False, target_attention_only: bool = False, min_rank: int = 4, target_modules: list[str] | None = None, **kwargs: Any)`
:   MGIE-SVD Compressor (Maximum Information Entropy SVD).

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.