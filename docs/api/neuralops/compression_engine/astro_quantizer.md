Module neuralops.compression_engine.astro_quantizer
===================================================
AstroQuantizer: Dynamic Uniform Range Quantization Engine with Straight-Through Estimator (STE)
and Fisher-Adaptive Mixed-Precision Bit Allocation. Registered as 'astro' and 'astro_quantizer'.

Scientific Novelty:
- Mixed-precision dynamic bit allocation driven by Empirical Fisher Information / Output Activation Energy.
- Optimal integer budget allocation: Channels with higher Fisher sensitivity receive higher bit-widths (up to max_bits),
  while less sensitive channels are quantized with lower bit-widths (down to min_bits), satisfying an exact average bit budget.
- Straight-Through Estimator (STE) gradients flow seamlessly across channels with heterogeneous bit allocations.
- Full robustness across LoRALinear wrappers, Conv1D layers (GPT-2), and multi-architecture models (DistilBERT, RoBERTa, GPT-2, Qwen3.5).

Functions
---------

`allocate_fisher_bits(importance: torch.Tensor, target_avg_bits: float = 6.0, min_bits: int = 2, max_bits: int = 8) ‑> torch.Tensor`
:   Allocates per-channel bit-widths b_i in [min_bits, max_bits] maximizing weighted importance
    such that sum(b_i) / N == target_avg_bits (within +/- 1/N tolerance).
    
    Formulation:
      max sum_i (importance_i * b_i)
      s.t. sum_i b_i <= round(N * target_avg_bits)
           min_bits <= b_i <= max_bits
    
    Solved via greedy knapsack: sort channels descending by importance, assign max_bits
    to top channels until budget is exhausted, then assign min_bits to remainder.

`astro_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Astro compressor factory function.

`quantize_dequantize_tensor(tensor: torch.Tensor, bits: int = 8, symmetric: bool = True, per_channel: bool = False, use_ste: bool = False, per_channel_bits: torch.Tensor | None = None) ‑> torch.Tensor`
:   Uniform quantization and de-quantization supporting:
    - Fixed bitwidth (scalar bits)
    - Heterogeneous per-channel bitwidths (per_channel_bits)
    - Symmetric or asymmetric ranges
    - Straight-Through Estimator (STE) autograd graph

Classes
-------

`AstroLinear(in_features: int, out_features: int, bias: bool = True, bits: int = 8, symmetric: bool = True, per_channel: bool = False, per_channel_bits: torch.Tensor | None = None)`
:   Linear layer using Astro uniform quantization with STE backpropagation support.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, bits: int = 8, symmetric: bool = True, per_channel: bool = False, per_channel_bits: torch.Tensor | None = None) ‑> neuralops.compression_engine.astro_quantizer.AstroLinear`
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

`AstroQuantizer(model: nn.Module, bits: int = 8, symmetric: bool = True, per_channel: bool = False, use_ste: bool = False, max_error: float = 0.2, fallback_to_original: bool = True, protect_critical: bool = True, target_modules: list[str] | None = None, target_attention_only: bool = False, use_fisher_adaptive_bits: bool = False, target_avg_bits: float = 6.0, min_bits: int = 2, max_bits: int = 8, fisher_beta: float = 0.5, **kwargs: Any)`
:   AstroQuantizer: Uniform min-max / symmetric quantization engine with Fisher-adaptive bit allocation.
    
    Production hardening features:
    - Support for LoRALinear wrappers, nn.Linear, and Conv1D (GPT-2).
    - Novel Fisher-Adaptive Bit Allocation allocating mixed-precision bit-widths [min_bits, max_bits] per output channel.
    - target_attention_only parameter skipping FFN/MLP and preserving classification heads.
    - Robust config handling (_cfg_get) supporting None, dict, and namespace objects.
    - Fallback freezing: parameters in any fallback layer are frozen (requires_grad=False).
    - Full FP32 quantization calculation with NaN/Inf guards.
    - Straight-Through Estimator (STE) autograd flow.
    - Transparent metric tracking: actual_average_bits, effective_compression_rate, and theoretical_vram_saving.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Quantize all target linear/Conv1D layers in the model and return (compressed_model, stats).

`STEFakeQuantize(*args, **kwargs)`
:   Straight-Through Estimator (STE) fake-quantization autograd function.
    Clamps inputs to [qmin, qmax] in discrete integer space, de-quantizes back,
    and passes gradients through uninhibited across all channels.

    ### Ancestors (in MRO)

    * torch.autograd.function.Function
    * torch.autograd.function._SingleLevelFunction
    * torch._C._FunctionBase
    * torch.autograd.function.FunctionCtx
    * torch.autograd.function._HookMixin

    ### Static methods

    `backward(ctx: Any, grad_output: torch.Tensor) ‑> tuple[torch.Tensor | None, ...]`
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

    `forward(ctx: Any, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, qmin: float | torch.Tensor, qmax: float | torch.Tensor) ‑> torch.Tensor`
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