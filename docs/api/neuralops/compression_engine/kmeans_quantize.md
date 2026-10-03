Module neuralops.compression_engine.kmeans_quantize
===================================================
KMeans-based weight clustering quantizer with Straight-Through Estimator.

Registered as 'kmeans_class', 'kmeans_quantize_class', 'kmeans_compressor_class',
              'kmeans', 'kmeans_quantize', 'kmeans_factory', 'kmeans_quantize_factory'.

Functions
---------

`fast_kmeans_1d(tensor: torch.Tensor, n_clusters: int = 256, max_iters: int = 20, tol: float = 0.0001, seed: int = 51966) ‑> tuple[torch.Tensor, torch.Tensor]`
:   1D Lloyd's algorithm with quantile init.
    
    KM-FIX-21: uses a deterministic seed for tie-breaking so results are
    reproducible. Returns (centroids, indices) where indices has the shape of
    `tensor` and its dtype is chosen to be the smallest that fits n_clusters.

`kmeans_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   

Classes
-------

`KMeansCompressor(model: nn.Module | None = None, n_clusters: int = 256, bits: int | None = None, max_error: float = 0.35, target_attention_only: bool = False, fallback_to_original: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = False, skip_should_compress_filter: bool = False, min_layer_dim: int = 8, kmeans_seed: int = 51966, cfg: Any | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`KMeansLinear(linear: nn.Module, n_clusters: int = 256, lora_r: int = 0, lora_alpha: float = 1.0, kmeans_seed: int = 51966)`
:   Linear layer with weight matrix quantized via 1D k-means codebook.
    
    Storage:
      - centroids:  (C,)         fp16/fp32
      - indices:    (out, in)    int8/int16/int32  (smallest that fits C)
      - bias:       (out,)       fp16/fp32
      - lora_A/B:   (r, in) / (out, r)   optional
    
    Forward:
      W_hat = centroids[indices]
      y = F.linear(x, W_hat) + bias + scaling · (x @ A^T) @ B^T
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `is_custom_linear: bool`
    :   The type of the None singleton.

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

    `get_weight(self) ‑> torch.Tensor`
    :

    `storage_bytes(self) ‑> dict[str, int]`
    :   KM-FIX-07/22: honest storage accounting.

    `train(self, mode: bool = True) ‑> neuralops.compression_engine.kmeans_quantize.KMeansLinear`
    :   Set the module in training mode.
        
        This has an effect only on certain modules. See the documentation of
        particular modules for details of their behaviors in training/evaluation
        mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.
        
        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.
        
        Returns:
            Module: self

`KMeansSTEFunction(*args, **kwargs)`
:   STE for codebook lookup.
    
    Forward:  out = F.linear(x, centroids[indices], bias)
    Backward: gradient to centroids via scatter-add; gradient to x flows
              through the same effective W.

    ### Ancestors (in MRO)

    * torch.autograd.function.Function
    * torch.autograd.function._SingleLevelFunction
    * torch._C._FunctionBase
    * torch.autograd.function.FunctionCtx
    * torch.autograd.function._HookMixin

    ### Static methods

    `backward(ctx, grad_output)`
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

    `forward(ctx, centroids, indices, x, bias)`
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