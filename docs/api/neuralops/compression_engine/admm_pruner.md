Module neuralops.compression_engine.admm_pruner
===============================================
ADMM Pruner: Alternating Direction Method of Multipliers for Structured & Semi-Structured Pruning
with Layer-Wise Reconstruction Distillation, STE, and Dynamic Rho.
Registered in COMPRESSOR_REGISTRY as 'admm_pruner' and 'admm_pruner_distill'.

Functions
---------

`admm_pruner_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Factory for ADMM pruner.

Classes
-------

`ADMMMaskedLinear(in_features: int, out_features: int, bias: bool = True, dtype: torch.dtype = torch.float32, device: torch.device | None = None)`
:   Linear layer with continuous weight + binary mask and STE gradient.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, mask: torch.Tensor) ‑> neuralops.compression_engine.admm_pruner.ADMMMaskedLinear`
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

`ADMMPruner(model: nn.Module, sparsity: float = 0.5, rho: float = 0.001, max_iter: int = 6, structured_24: bool = True, block_size: tuple[int, int] = (4, 4), dynamic_rho: bool = True, use_ste: bool = True, recalibrate_ln: bool = True, max_error: float = 0.45, fallback_to_original: bool = True, protect_critical: bool = True, target_attention_only: bool = False, target_modules: list[str] | None = None, lambda_distill: float = 0.1, distill_batch_size: int = 32, distill_use_activation: bool = False, distill_lr: float = 0.1, distill_steps: int = 2, calibration_inputs: dict[str, torch.Tensor] | None = None, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Production-hardened ADMM Pruner with Layer-Wise Reconstruction Distillation,
    2:4 semi-structured sparsity, STE gradients, and dynamic rho scheduling.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

`ADMMSTEPrune(*args, **kwargs)`
:   Straight-Through Estimator for ADMM sparse weights.
    Forward: y = weight * mask (with dtype-aligned mask).
    Backward: gradient flows through the identity to weight; mask has no grad.

    ### Ancestors (in MRO)

    * torch.autograd.function.Function
    * torch.autograd.function._SingleLevelFunction
    * torch._C._FunctionBase
    * torch.autograd.function.FunctionCtx
    * torch.autograd.function._HookMixin

    ### Static methods

    `backward(ctx, grad_output: torch.Tensor)`
    :   Define a formula for differentiating the operation with backward mode automatic differentiation.
        
        This function is to be overridden by all subclasses.
        (Defining this function is equivalent to defining the ``vjp`` function.)
        
        It must accept a context :attr:`ctx` as the first argument, followed by
        as many outputs as the :func:`forward` returned (None will be passed in
        for non tensor outputs of the forward function),
        and it should return as many tensors, as there were inputs to
        :func:`forward`. Each argument is the gradient w.r.t the given output,
        and each returned value should be the gradient w.r.t. the
        corresponding input. If an input is not a Tensor or is a Tensor not
        requiring grads, you can just pass None as a gradient for that input.
        
        The context can be used to retrieve tensors saved during the forward
        pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
        of booleans representing whether each input needs gradient. E.g.,
        :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
        first input to :func:`forward` needs gradient computed w.r.t. the
        output.

    `forward(ctx, weight: torch.Tensor, mask: torch.Tensor) ‑> torch.Tensor`
    :   Define the forward of the custom autograd Function.
        
        This function is to be overridden by all subclasses.
        There are two ways to define forward:
        
        Usage 1 (Combined forward and ctx)::
        
            @staticmethod
            def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
                pass
        
        - It must accept a context ctx as the first argument, followed by any
          number of arguments (tensors or other types).
        - See :ref:`combining-forward-context` for more details
        
        Usage 2 (Separate forward and ctx)::
        
            @staticmethod
            def forward(*args: Any, **kwargs: Any) -> Any:
                pass
        
        
            @staticmethod
            def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
                pass
        
        - The forward no longer accepts a ctx argument.
        - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
          staticmethod to handle setting up the ``ctx`` object.
          ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
          to the forward.
        - See :ref:`extending-autograd` for more details
        
        The context can be used to store arbitrary data that can be then
        retrieved during the backward pass. Tensors should not be stored
        directly on `ctx` (though this is not currently enforced for
        backward compatibility). Instead, tensors should be saved either with
        :func:`ctx.save_for_backward` if they are intended to be used in
        ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
        if they are intended to be used for in ``jvp``.