Module neuralops.utils.compat
=============================
Family-aware, opt-in compatibility helpers for attention modules.

Bug fixes (2026-09-25):
- Fix CP-01: get_recovery_params now preserves SVDLinear U/V factor matrices
  during the parameter-budget pruning step, even if their numel exceeds the
  32768 threshold. Previously, recovery was reduced to bias-only training,
  crippling recovery on compressed models.
- Fix CP-02: Parameter-budget pruning now uses priority-based allocation:
  (1) protected params (LoRA/gates/classifier), (2) SVDLinear U/V factors,
  (3) small params (1D, or numel <= 32768), (4) other large 2D params.
- Fix CP-03: The 5M-rank-clamp now correctly preserves the LoRA's alpha/r
  scaling by adjusting both lora_rank and lora_alpha together.
- Fix CP-04: Better diagnostics: log how many params were kept per priority
  bucket so recovery behavior is observable.

Functions
---------

`causal_mask_from_attention_mask(attention_mask: torch.Tensor | None, adapter: ModelAdapter)`
:   Return the mask without discarding decoder causal visibility semantics.

`convert_mask(mask: torch.Tensor | None, adapter: ModelAdapter, expects_bidirectional: bool = False) ‑> torch.Tensor | None`
:   

`fp32_linalg(op, *tensors: torch.Tensor, **kwargs: Any)`
:   

`get_dynamic_compression_rank(in_features: int, out_features: int, model_or_family: Any = None, default_rank: int = 16, comp_config: dict | None = None) ‑> int`
:   Compute architecture-aware dynamic compression rank.
    
    For Qwen3.5: rank = min(256, max(128, min(in_features, out_features) // 4))
    For other families: preserves default fixed/configured rank.

`get_recovery_params(model: nn.Module) ‑> list[torch.nn.parameter.Parameter]`
:   Select recovery parameters for LoRA / SVD / injected attention modules.
    
    Fix CP-01: SVDLinear U/V factor matrices are now preserved during the
    parameter budget pruning step. Previously, their numel (e.g. 64 * 768 =
    49152) exceeded the 32768 threshold and they were pruned, reducing recovery
    to bias-only training.
    
    Fix CP-02: Pruning now uses priority-based allocation:
        Priority 0 (highest): protected params (LoRA / gates / classifier / alpha / beta)
        Priority 1: SVDLinear U/V factors and factorized parameters
        Priority 2: small params (1D, or numel <= 32768)
        Priority 3 (lowest): other large 2D params

`init_output_scale(module: nn.Module, name: str = 'output_scale') ‑> None`
:   Register output_scale learnable parameter if not already present.

`install_projection_aliases(module: nn.Module) ‑> torch.nn.modules.module.Module`
:   Install bidirectional attribute aliases across canonical projection names.

`is_decoder_family(family: str) ‑> bool`
:   Check whether a family (canonical or raw) is a causal decoder.
    
    Fix CP-00: Now uses the canonical CAUSAL_FAMILIES set from model_family,
    so all GPT variants (gpt2, llama, mistral, ...) are correctly classified.

`is_decoder_projection_name(name: str) ‑> bool`
:   Whether a parameter name denotes a decoder attention projection.

`maybe_apply_rotary(q: torch.Tensor, k: torch.Tensor, adapter: ModelAdapter, rotary_emb: Any, position_ids: torch.Tensor | None)`
:   Apply a native rotary module when required; never silently drop RoPE.

`module_family_support(name: str) ‑> dict[str, str]`
:   

`repeat_kv(x: torch.Tensor, n_rep: int) ‑> torch.Tensor`
:   Expand grouped-query K/V heads: [B, Hkv, L, D] -> [B, H, L, D].

`should_compress_linear(model: Any, name: str, family: str | None = None) ‑> bool`
:   Family-aware check for whether a linear layer should be compressed.
    
    For qwen3_5: sensitive layers (attention, DeltaNet, GQA, MoE routers/experts,
    norms, RoPE, embeddings, heads) are protected; only ordinary FFN dense layers
    are compressed. For other families, safe linear layer compression is preserved.

`should_compress_qwen3_5(name: str) ‑> bool`
:   

`wrap_for_family(module: nn.Module, model_family: str | None = None) ‑> torch.nn.modules.module.Module`
:   

Classes
-------

`FamilyModuleWrapper(module: nn.Module, family: str)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing them to be nested in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
    
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will also have their
    parameters converted when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, *args: Any, **kwargs: Any) ‑> Callable[..., typing.Any]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.