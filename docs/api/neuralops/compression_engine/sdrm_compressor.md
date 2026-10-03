Module neuralops.compression_engine.sdrm_compressor
===================================================
SDRM: Scaled Dimension Rank Modulation.
Assigns layer rank proportional to parameter budget for edge inference and factorizes
into low-rank SVDLinear layers to achieve physical parameter reduction.
Registered as 'sdrm', 'sdrm_factory', and 'sdrm_compressor_factory'.

Functions
---------

`sdrm_compressor_factory(model: Any, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   SDRM compressor factory returning (compressed_model, stats) when requested.

Classes
-------

`SDRMCompressor(model: nn.Module, total_param_ratio: float = 0.7, min_rank: int = 8, skip_attention_qkv: bool = False, target_attention_only: bool = False, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, target_modules: list[str] | None = None, max_error: float = 0.5, fallback_to_original: bool = True, min_numel: int = 16, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   SDRM: Scaled Dimension Rank Modulation.
        Assigns layer rank proportional to parameter budget for edge inference and factorizes
        target layers into low-rank SVDLinear modules with optional LoRA fine-tuning adapters.

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress linear layers using budget-proportional rank assignment.

    `compress_linear(self, layer: nn.Module, rank: int) ‑> neuralops.compression_engine.aa_svd_compressor.SVDLinear | None`
    :   Decompose a single linear or Conv1D layer into SVDLinear in FP32 with validation.