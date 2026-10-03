Module neuralops.utils.qwen_compat
==================================
Compatibility helpers for decoder-only Qwen attention blocks.

This module only exposes the names expected by legacy attention/compressor
implementations.  It does not replace Qwen's rotary or causal attention.

Functions
---------

`bypass_qwen_rope(kwargs: dict[str, Any], module: Any) ‑> dict[str, typing.Any]`
:   Prepare kwargs for a module that does not implement RoPE.
    
    This does not modify the base model or its rotary module. The marker is
    consumed by wrappers and the positional tensors are omitted only from the
    injected replacement call.

`fp32_linalg(fn, *tensors: torch.Tensor, **kwargs: Any)`
:   Run QR/SVD/eigh-style operations in fp32 and restore output dtype.

`install_qwen_aliases(model: nn.Module) ‑> torch.nn.modules.module.Module`
:   Install non-structural projection aliases on every Qwen attention block.

`is_architecture_error(exc: BaseException) ‑> bool`
:   

`is_dtype_error(exc: BaseException) ‑> bool`
:   

`is_qwen_attention(module: nn.Module) ‑> bool`
:   Detect a native Qwen attention block without importing transformers.

`is_qwen_model(model: nn.Module) ‑> bool`
:   

`qwen_attention_mask(mask: torch.Tensor | None, *, causal: bool = True, expects_bidirectional: bool = False) ‑> torch.Tensor | None`
:   Adapt masks only for modules that explicitly request another contract.
    
    Native Qwen attention keeps its prepared causal 4-D additive mask. A
    replacement that declares ``expects_bidirectional`` receives a boolean
    visibility mask; other callers receive the original object unchanged.

`qwen_config_proxy(model_or_config: Any) ‑> neuralops.utils.qwen_compat.QwenConfigProxy`
:   

`qwen_module_policy(module_name: str) ‑> str`
:   Classify modules whose semantics cannot be promised for causal RoPE Qwen.

`qwen_projection(module: nn.Module, legacy: str, native: str) ‑> torch.nn.modules.module.Module | None`
:   Return a projection under either its legacy or native spelling.

Classes
-------

`QwenConfigProxy(source: Any, hidden_size: int, num_attention_heads: int, head_dim: int, num_hidden_layers: int)`
:   Small, read-through config view with the common attention fields.

    ### Instance variables

    `head_dim: int`
    :   The type of the None singleton.

    `hidden_size: int`
    :   The type of the None singleton.

    `num_attention_heads: int`
    :   The type of the None singleton.

    `num_hidden_layers: int`
    :   The type of the None singleton.

    `source: Any`
    :   The type of the None singleton.

`QwenIncompatibleError(*args, **kwargs)`
:   Raised when a module has no truthful Qwen execution path.

    ### Ancestors (in MRO)

    * builtins.RuntimeError
    * builtins.Exception
    * builtins.BaseException

    ### Class variables

    `reason`
    :   The type of the None singleton.