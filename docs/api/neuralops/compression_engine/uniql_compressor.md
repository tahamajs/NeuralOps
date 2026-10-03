Module neuralops.compression_engine.uniql_compressor
====================================================
UniQL: Joint low-rank + uniform quantization compressor.

Registered as 'uniql', 'uniql_compressor', 'uniql_factory'.

Functions
---------

`uniql_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   UniQL factory.

Classes
-------

`UniQLCompressor(model: nn.Module, rank: int = 64, bits: int = 8, max_error: float = 0.2, max_retries: int = 3, fallback_to_original: bool = True, target_attention_only: bool = False, quantize_bias: bool = False, **kwargs: Any)`
:   UniQL: Joint Truncated SVD + Uniform Quantization compressor.
    
    Unlike the naive "copy back" variant, this compressor installs a real
    UniQLinear module that stores int8 U, S, Vt buffers. Parameter count
    and memory footprint decrease by roughly the SVD factor
    (out + in) * r / (out * in).

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`UniQLinear(in_features: int, out_features: int, rank: int, U_q: torch.Tensor, S_q: torch.Tensor, Vt_q: torch.Tensor, scale_u: torch.Tensor, scale_s: torch.Tensor, scale_v: torch.Tensor, bias: torch.Tensor | None = None, is_conv1d: bool = False, dtype: torch.dtype = torch.float32, device: torch.device | None = None)`
:   Replaces a Linear (or Conv1D) with a stored factorization:
    
        W ≈ U @ diag(S) @ Vt
        where U  : (out, r) int8, scale_u  (out,)
              S  : (r,)    int8, scale_s  scalar-per-rank
              Vt : (r, in) int8, scale_v  (in,)
    
    Forward reconstructs W_hat on the fly and applies F.linear / F.linear with
    transposed Conv1D weight. Numerically this is exactly the same
    reconstruction as the original UniQL, but now the storage is genuinely
    reduced from out*in fp16 to (out + in) * r int8 plus small scale vectors.
    
    Original bias (if any) is stored as-is unless quantize_bias=True.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed weight (out, in). Conv1D callers should transpose.

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