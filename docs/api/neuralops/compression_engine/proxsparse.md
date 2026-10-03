Module neuralops.compression_engine.proxsparse
==============================================
ProxSparse: Proximal Gradient Regularized Structured & Semi-Structured Sparsity.

Registered in COMPRESSOR_REGISTRY as 'proxsparse', 'proxsparse_pruner',
'proxsparse_compressor', 'proxsparse_factory', ...

Mathematical Formulation:
    min_W  0.5 * ||W - W_0||_F^2 + lambda * ||W||_1 + I_S(W)

where:
  - W_0 is the original (pretrained) weight,
  - ||.||_1 is the L1 regularizer,
  - I_S(W) is the indicator of the sparse constraint set S (percentile,
    block, or 2:4 semi-structured).

Iterative Proximal Gradient (ISTA with projection):
    for t = 1, ..., steps:
        grad_W    = W^(t) - W_0                  # gradient of smooth part
        W_grad    = W^(t) - eta * grad_W         # gradient descent step
        W_prox    = S_{eta*lambda}(W_grad)       # soft-thresholding
        W^(t+1)   = Pi_S(W_prox)                 # projection to S

where:
  - S_a(x) = sign(x) * max(|x| - a, 0)  (soft-thresholding)
  - Pi_S is the hard projection (percentile/2:4/block top-k)

Functions
---------

`proxsparse_factory(model: Any, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   

Classes
-------

`ProxSparsePrunedLoRALinear(in_features: int, out_features: int, pruned_weight: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Frozen pruned base weight + trainable LoRA adapters.
    
    NOTE: `weight` is the ALREADY-PRUNED weight matrix. Forward does not
    re-apply the mask. `mask` is kept only as a compact bool buffer for
    downstream analysis / merge operations.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, pruned_weight: torch.Tensor, mask: torch.Tensor, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.proxsparse.ProxSparsePrunedLoRALinear`
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

`ProxSparsePruner(model: nn.Module, sparsity: float = 0.5, lambda_prox: float = 0.01, steps: int = 5, learning_rate: float = 0.5, structured_24: bool = False, block_size: tuple[int, int] = (1, 1), skip_attention_qkv: bool = True, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, max_error: float | None = None, fallback_to_original: bool = True, min_numel: int = 16, target_modules: list[str] | None = None, target_attention_only: bool = False, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   ProxSparse: proximal gradient sparsity with structured / 2:4 / percentile
    constraint, optional LoRA fine-tuning, and safe fallback.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Return (compressed_model, statistics).

`ProxSparseCompressor(model: nn.Module, sparsity: float = 0.5, lambda_prox: float = 0.01, steps: int = 5, learning_rate: float = 0.5, structured_24: bool = False, block_size: tuple[int, int] = (1, 1), skip_attention_qkv: bool = True, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, max_error: float | None = None, fallback_to_original: bool = True, min_numel: int = 16, target_modules: list[str] | None = None, target_attention_only: bool = False, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   ProxSparse: proximal gradient sparsity with structured / 2:4 / percentile
    constraint, optional LoRA fine-tuning, and safe fallback.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor