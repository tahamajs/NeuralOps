Module neuralops.compression_engine.roast_hashed_linear
=======================================================
ROAST: Hardware-Aware Weight Sharing with Tiling.
Reference: 'ROAST: A Hardware-Aware Weight Sharing Scheme for Efficient Neural Networks'
Registered in COMPRESSOR_REGISTRY as 'roast', 'roast_compressor', and 'roast_factory'.

Functions
---------

`roast_factory(model: Any, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   ROAST compressor factory function returning (compressed_model, stats) when requested.

Classes
-------

`ROASTCompressor(model: nn.Module, n_buckets: int = 1024, tile_size: int = 32, skip_attention_qkv: bool = True, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, target_modules: list[str] | None = None, target_attention_only: bool = False, max_error: float = 0.5, fallback_to_original: bool = True, min_numel: int = 16, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Replace eligible linear and Conv1D layers with ROASTLinear (hardware-aware weight sharing).

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress the target modules using ROAST tiling with LoRA support.

`ROASTLinear(in_features: int, out_features: int, n_buckets: int = 1024, tile_size: int = 32, bias: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Linear layer with weight sharing based on tiles (blocks) rather than
    individual scalar weights. Each tile of size (tile_size, tile_size) shares
    a single weight from a small table, with optional trainable LoRA adapters.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, n_buckets: int = 1024, tile_size: int = 32, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.roast_hashed_linear.ROASTLinear`
    :   Initialize from a pretrained nn.Linear, Conv1D, or LoRALinear base_layer.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix (out_features, in_features) with evaluation caching.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Vectorized forward pass without per-element python loops.