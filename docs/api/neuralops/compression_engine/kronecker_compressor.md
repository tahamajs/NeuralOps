Module neuralops.compression_engine.kronecker_compressor
========================================================
Kronecker Product Compressor: replaces nn.Linear and Conv1D layers with KroneckerLinear.
Registered as 'kronecker_class', 'kronecker_compressor_class', and (factory)
'kronecker', 'kronecker_compressor', 'kronecker_factory'.

Functions
---------

`kronecker_factory(model: nn.Module, comp_config: Any | None = None, rank: int | None = None, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   Kronecker compressor factory.

Classes
-------

`KroneckerCompressor(model: nn.Module | None = None, rank: int | None = None, max_error: float = 0.8, fallback_to_original: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, target_attention_only: bool = False, skip_attention_qkv: bool = False, skip_should_compress_filter: bool = False, min_layer_dim: int = 8, cfg: Any | None = None, target_layers: list[str] | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.