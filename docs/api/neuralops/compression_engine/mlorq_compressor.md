Module neuralops.compression_engine.mlorq_compressor
====================================================
MLoRQ: Multi-Layer Optimal Rank & Quantisation under a Total Memory Budget.

Registered as 'mlorq', 'mlorq_compressor', 'mlorq_factory'.

This version stores U, S, Vt as PACKED INT tensors (bits ∈ {2,4,8}), so the
memory budget formula `(out*r + r*in + r) * bits` is exact. The forward pass
dequantizes on the fly and never materializes the dense weight.

Functions
---------

`mlorq_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   

Classes
-------

`MLoRQCompressor(model: nn.Module, total_memory_budget_mb: float = 500, bits_candidates: list[int] | None = None, max_error: float = 0.5, fallback_to_original: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = False, target_attention_only: bool = False, min_rank: int = 4, target_modules: list[str] | None = None, **kwargs: Any)`
:   MLoRQ Compressor: Multi-Layer Optimal Rank and Quantization under Memory Budget.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`MLoRQLoRALinear(in_features: int, out_features: int, rank: int, bits: int, U_packed: torch.Tensor, U_scale: torch.Tensor, U_shape: torch.Size, S_packed: torch.Tensor, S_scale: torch.Tensor, S_shape: torch.Size, Vt_packed: torch.Tensor, Vt_scale: torch.Tensor, Vt_shape: torch.Size, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Low-rank quantized linear layer:
    
        y = (x @ Vt) * S @ U^T + bias + LoRA(x)
    
    where U, S, Vt are stored as PACKED int8 tensors with per-tensor scales.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_linear(linear: nn.Module, U: torch.Tensor, S: torch.Tensor, Vt: torch.Tensor, bits: int, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.mlorq_compressor.MLoRQLoRALinear`
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

    `storage_bytes(self) ‑> int`
    :