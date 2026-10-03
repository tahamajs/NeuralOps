Module neuralops.compression_engine.coopq_quantizer
===================================================
CoopQ: Cooperative Game-Theoretic Bit Allocation Quantizer with STE and Per-Channel Asymmetric Calibration.
Registered in COMPRESSOR_REGISTRY as 'coopq'.

Mathematical Formulation:
1. Min Bit-Width Floor Enforcement:
   $$b_l \ge b_{\min} = 4, \quad \forall l \in \{1, \dots, L\}$$
2. Per-Channel Asymmetric Scale & Zero-Point Calibration:
   $$\text{scale}_i = \max\left( \frac{\max_j W_{ij} - \min_j W_{ij}}{2^b - 1}, 10^{-8} \right), \quad z_i = \operatorname{clamp}\left( \left\lfloor -\frac{\min_j W_{ij}}{\text{scale}_i} \right\rceil, 0, 2^b - 1 \right)$$
   $$\widehat{W}_{ij} = \left( \operatorname{clamp}\left( \left\lfloor \frac{W_{ij}}{\text{scale}_i} \right\rceil + z_i, 0, 2^b - 1 \right) - z_i \right) \cdot \text{scale}_i$$
3. Straight-Through Estimator (STE) Gradient Backpropagation:
   $$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial \widehat{W}}$$

Functions
---------

`coopq_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   CoopQ compressor factory function.

`quantize_dequantize_coopq(tensor: torch.Tensor, bits: int = 4, symmetric: bool = False, per_channel: bool = True, use_ste: bool = False) ‑> torch.Tensor`
:   Calibrates per-channel / tensor-wide scale and zero-point for INT(bits) and applies
    symmetric or asymmetric uniform quantization with optional STE.
    Enforces minimum bit-width floor b >= 4.

Classes
-------

`CoopQLinear(in_features: int, out_features: int, bias: bool = True, bits: int = 4, symmetric: bool = False, per_channel: bool = True, lora_r: int = 0, lora_alpha: float = 1.0)`
:   CoopQ Linear layer with per-channel asymmetric quantization and STE backpropagation.
    Supports both standard nn.Linear and Conv1D weight representations.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, bits: int = 4, symmetric: bool = False, per_channel: bool = True, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.coopq_quantizer.CoopQLinear`
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

`CoopQQuantizer(model: nn.Module, bits: int = 4, min_bits: int = 4, max_bits: int = 8, symmetric: bool = False, per_channel: bool = True, use_ste: bool = True, recalibrate_ln: bool = True, target_attention_only: bool = False, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   CoopQ: Cooperative Game-Theoretic Bit Allocation Quantization.
    Allocates discrete bit-widths (>= 4 bits) across transformer layers
    proportional to layer sensitivity (Frobenius energy / variance), guaranteeing
    a target mean bit-budget under strict distortion minimization.
    
    Args:
        model: PyTorch model.
        bits: Target average bit-width (default 4).
        min_bits: Minimum allowable bit-width per layer (floor enforced at >= 4).
        max_bits: Maximum allowable bit-width per layer (default 8).
        symmetric: Whether to use symmetric quantization (default False for asymmetric calibration).
        per_channel: Whether to use per-channel scale factors (default True).
        use_ste: Whether to substitute nn.Linear with CoopQLinear for QAT (default True).
        target_attention_only: If True, only quantize attention projection layers (skip FFN/MLP).
        cfg: Optional configuration dictionary, namespace, or object.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Quantize linear layers with layer-specific bit allocation (>= 4 bits) and per-channel asymmetric calibration.

`CoopQSTEFakeQuantize(*args, **kwargs)`
:   Straight-Through Estimator (STE) fake-quantization autograd function for CoopQ.
    Forwards per-channel quantized-dequantized weights and passes gradients through on backward pass.

    ### Ancestors (in MRO)

    * torch.autograd.function.Function
    * torch.autograd.function._SingleLevelFunction
    * torch._C._FunctionBase
    * torch.autograd.function.FunctionCtx
    * torch.autograd.function._HookMixin

    ### Static methods

    `backward(ctx: Any, grad_output: torch.Tensor) ‑> tuple[torch.Tensor, None, None, None, None]`
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

    `forward(ctx: Any, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, qmin: float, qmax: float) ‑> torch.Tensor`
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