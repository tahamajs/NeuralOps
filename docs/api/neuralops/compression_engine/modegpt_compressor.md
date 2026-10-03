Module neuralops.compression_engine.modegpt_compressor
======================================================
MoDeGPT: Modular Decomposition of Transformer Blocks — Production-Hardened (v2).

Decomposes pairs of linear projections sharing the same input dimension
(e.g., Q & K, Q & V, Gate & Up) into shared low-rank bases:

    W1 ≈ U1 @ V_shared      U1 ∈ R^{out1 × r}, V_shared ∈ R^{r × in}
    W2 ≈ U2 @ V_shared      U2 ∈ R^{out2 × r}

The shared factor V_shared is stored ONCE per pair. This yields real parameter
and storage reduction for the pair as a whole.

For unpaired layers, both U and V are stored per-layer.

Functions
---------

`modegpt_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   

Classes
-------

`MoDeGPTCompressor(model: nn.Module, rank: int = 64, method: str = 'svd', max_error: float = 0.5, fallback_to_original: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = False, target_attention_only: bool = False, min_rank: int = 4, target_modules: list[str] | None = None, **kwargs: Any)`
:   MoDeGPT: Modular Decomposition of Transformer Projection Blocks.
    
    Shares a low-rank input basis V across paired projections sharing the same
    input dimension. This yields real storage reduction for the pair:
    
        Original: out1·in + out2·in        = (out1+out2)·in
        MoDeGPT : r·in + out1·r + out2·r   = r·(in + out1 + out2)
    
    Storage reduction when r < in·(out1+out2) / (in + out1 + out2).

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`MoDeGPTLowRankLinear(in_features: int, out_features: int, rank: int, U: torch.Tensor, V: torch.Tensor | None = None, shared_V: torch.Tensor | None = None, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Low-rank linear layer.
    
        y = x @ V^T @ U^T + bias
          = (x @ V^T) @ U^T + bias
        where U ∈ R^{out × r}, V ∈ R^{r × in}
    
    If `shared_V` is provided, this module uses the external tensor instead of
    its own `V` parameter. This enables sharing the input basis across paired
    projections (Q/K, Gate/Up, ...), giving real compression for the pair.
    
    Optional LoRA adapters (lora_A, lora_B) are trained in addition to the
    frozen low-rank base.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed (out, in) weight matrix.

    ### Methods

    `extra_repr(self) ‑> str`
    :   Return the extra representation of the module.
        
        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.