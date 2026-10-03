Module neuralops.attentions.fused_svd_kernel
============================================
Fused SVD-Linear Layer with SRAM-Resident RMSNorm & Dual-Projection.

RESEARCH FIX: Eliminates GPU High-Bandwidth Memory (HBM) roundtrip latency by fusing:
    1. Down-projection: Z = X @ V^T  (hidden_dim -> rank)
    2. In-SRAM RMSNorm: Z_norm = Z * rsqrt(mean(Z^2) + eps) * weight
    3. Up-projection:   Y = Z_norm @ U^T + bias (rank -> out_dim)

The intermediate activation matrix Z \in R^{B x L x r} is kept strictly inside
GPU Shared Memory (SRAM) registers and never materialized to DRAM/HBM.
Includes Triton JIT accelerated kernel with CUDA tensor-core blocking and
PyTorch autograd function with fallback.

Functions
---------

`fused_svd_kernel_factory(config: Any = None, **kwargs: Any) ‑> torch.nn.modules.module.Module`
:   Factory creating FusedSVDKernelAttention with fused SVD kernel execution.

`fused_svd_linear(x: torch.Tensor, V: torch.Tensor, U: torch.Tensor, norm_weight: torch.Tensor | None = None, bias: torch.Tensor | None = None, eps: float = 1e-06, use_rms_norm: bool = False) ‑> torch.Tensor`
:   Functional interface for Fused SVD-Linear with optional in-SRAM RMSNorm.
    
    Args:
        x: Input tensor of shape (..., in_features).
        V: Down-projection matrix of shape (rank, in_features).
        U: Up-projection matrix of shape (out_features, rank).
        norm_weight: Optional RMSNorm affine scale vector of shape (rank,).
        bias: Optional bias vector of shape (out_features,).
        eps: Small epsilon for RMSNorm numerical stability.
        use_rms_norm: Whether to apply in-SRAM RMSNorm on intermediate rank activations.
    
    Returns:
        Output tensor of shape (..., out_features).

`inject_fused_svd_kernel(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any) ‑> Any | None`
:   Safe injection hook for fused SVD kernel attention.

Classes
-------

`FusedSVDKernelAttention(config: Any | None = None, rank: int | None = None, old_attn: nn.Module | None = None, use_fused_kernel: bool = True, use_rms_norm: bool = False, dropout: float = 0.1, **kwargs: Any)`
:   Fused SVD-Kernel Linear Attention Module.
    Production-hardened across DistilBERT, RoBERTa/BERT nested, GPT-2 Conv1D, and Qwen/LLaMA.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Instance variables

    `key: FusedSVDLinear`
    :

    `query: FusedSVDLinear`
    :

    `value: FusedSVDLinear`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Multi-architecture weight copying supporting GPT-2 Conv1D, RoBERTa nested, and flat projections.

`FusedSVDLinear(in_features: int, out_features: int, rank: int = 32, bias: bool = True, use_rms_norm: bool = False, eps: float = 1e-06, dtype: torch.dtype = torch.float32, device: torch.device | None = None)`
:   RESEARCH SOTA: Fused SVD-Linear Module.
    Combines low-rank projection $Z = x V^T$, in-SRAM RMSNorm, and up-projection $Y = Z U^T + b$
    into a single kernel invocation, eliminating HBM roundtrips.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix for compatibility.

    ### Methods

    `decompose_from_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) ‑> None`
    :   Decompose dense weight matrix into symmetric SVD factors.

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

    `get_orthogonality_loss(self) ‑> torch.Tensor`
    :   # RESEARCH FIX: Computes Stiefel manifold penalty to prevent singular subspace collapse.

    `reset_parameters(self) ‑> None`
    :

    `set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) ‑> None`
    :   Set precomputed U and V low-rank factors.

`_FusedSVDLinearAutogradFunction(*args, **kwargs)`
:   Base class to create custom `autograd.Function`.
    
    To create a custom `autograd.Function`, subclass this class and implement
    the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
    op in the forward pass, call the class method ``apply``. Do not call
    :meth:`forward` directly.
    
    To ensure correctness and best performance, make sure you are calling the
    correct methods on ``ctx`` and validating your backward function using
    :func:`torch.autograd.gradcheck`.
    
    See :ref:`extending-autograd` for more details on how to use this class.
    
    Examples::
    
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> class Exp(Function):
        >>>     @staticmethod
        >>>     def forward(ctx, i):
        >>>         result = i.exp()
        >>>         ctx.save_for_backward(result)
        >>>         return result
        >>>
        >>>     @staticmethod
        >>>     def backward(ctx, grad_output):
        >>>         result, = ctx.saved_tensors
        >>>         return grad_output * result
        >>>
        >>> # Use it by calling the apply method:
        >>> # xdoctest: +SKIP
        >>> output = Exp.apply(input)

    ### Ancestors (in MRO)

    * torch.autograd.function.Function
    * torch.autograd.function._SingleLevelFunction
    * torch._C._FunctionBase
    * torch.autograd.function.FunctionCtx
    * torch.autograd.function._HookMixin

    ### Class variables

    `is_custom_attention: bool`
    :   Autograd Function dispatching Triton JIT fused kernel on CUDA GPUs
        with exact analytical backward gradient computation.

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

    `forward(ctx: Any, x: torch.Tensor, V: torch.Tensor, U: torch.Tensor, norm_weight: torch.Tensor | None = None, bias: torch.Tensor | None = None, eps: float = 1e-06, use_rms_norm: bool = False) ‑> torch.Tensor`
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