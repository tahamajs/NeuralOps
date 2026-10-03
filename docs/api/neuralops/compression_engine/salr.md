Module neuralops.compression_engine.salr
========================================
SALR: Structure-Aware / Sparse Low-Rank compression combining SVD with bottleneck pruning.
Registered in COMPRESSOR_REGISTRY as 'salr', 'salr_factory', and 'salr_compressor_factory'.

Functions
---------

`salr_factory(model: nn.Module, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   SALR factory function returning (compressed_model, stats) when requested.

Classes
-------

`SALRCompressor(model: nn.Module, rank: int = 64, sparsity: float = 0.5, min_rank: int = 8, skip_attention_qkv: bool = False, target_attention_only: bool = False, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, target_modules: list[str] | None = None, max_error: float = 0.5, fallback_to_original: bool = True, min_numel: int = 16, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   SALR Compressor: Structure-Aware Low-Rank and Sparse Pruning.
        
        Factorizes dense nn.Linear or Conv1D layers into symmetric low-rank factors
        (U * sqrt(Sigma), sqrt(Sigma) * V^T) and applies structured sparse pruning
        to trailing singular components in the bottleneck with LoRA adapters for recovery.

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress all linear layers in the model.

    `compress_linear(self, layer: nn.Module, rank: int) ‑> neuralops.compression_engine.salr.SALRLayer | None`
    :   Decompose a single linear or Conv1D layer into SALRLayer with validation.

`SALRLayer(in_features: int, out_features: int, rank: int, sparsity: float = 0.5, bias: bool = True, dtype: torch.dtype = torch.float32, device: torch.device | None = None, lora_r: int = 0, lora_alpha: float = 1.0)`
:   Structure-Aware Low-Rank + Sparse Layer (SALR).
    
    Combines truncated SVD rank reduction with sparse singular component selection
    and optional trainable LoRA fine-tuning adapters:
        y = (x @ (V * mask).T) @ U.T + bias + (lora_B @ lora_A @ x) * scaling
    where U: [out_features, rank], V: [rank, in_features], mask: [rank, 1].
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Instance variables

    `weight: torch.Tensor`
    :   Effective reconstructed weight matrix.

    ### Methods

    `extra_repr(self) ‑> str`
    :   Return the extra representation of the module.
        
        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Two-stage forward pass with sparse bottleneck mask and optional LoRA branch.

    `reset_parameters(self) ‑> None`
    :

    `set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) ‑> None`
    :