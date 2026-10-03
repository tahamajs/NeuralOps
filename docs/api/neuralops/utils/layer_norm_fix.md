Module neuralops.utils.layer_norm_fix
=====================================
Numerically Stable Layer Normalization and RMS Normalization Modules.

Eliminates FP16 variance underflow and sum-of-squares overflow by forcing all
mean, variance, and reduction accumulators into torch.float32 prior to reciprocal square root.

Functions
---------

`fix_layernorm_naming(model: nn.Module) ‑> None`
:   Fix LayerNorm parameter naming (e.g. HuggingFace gamma/beta to weight/bias).

`replace_layernorm_with_safe_norm(model: nn.Module) ‑> torch.nn.modules.module.Module`
:   Recursively replaces all standard PyTorch nn.LayerNorm instances in a model
    with FP32 accumulator SafeLayerNorm instances.

Classes
-------

`LayerNorm(normalized_shape: int | tuple[int, ...] | list[int] | torch.Size, eps: float = 1e-05, elementwise_affine: bool = True, bias: bool = True, device: torch.device | str | None = None, dtype: torch.dtype | None = None)`
:   Standard Layer Normalization with FP32 accumulator precision.
    
    Prevents FP16 numerical instability and variance underflow by computing
    mean and centered variance in `torch.float32`:
        $$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `extra_repr(self) ‑> str`
    :   Return the extra representation of the module.
        
        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Forward pass with FP32 accumulator computation.
        
        Args:
            x: Input tensor of shape `(..., *normalized_shape)`.
        
        Returns:
            Normalized tensor with matching input dtype and shape.

`SafeLayerNorm(normalized_shape: int | tuple[int, ...] | list[int] | torch.Size, eps: float = 1e-05, elementwise_affine: bool = True, bias: bool = True, device: torch.device | str | None = None, dtype: torch.dtype | None = None)`
:   Standard Layer Normalization with FP32 accumulator precision.
    
    Prevents FP16 numerical instability and variance underflow by computing
    mean and centered variance in `torch.float32`:
        $$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `extra_repr(self) ‑> str`
    :   Return the extra representation of the module.
        
        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Forward pass with FP32 accumulator computation.
        
        Args:
            x: Input tensor of shape `(..., *normalized_shape)`.
        
        Returns:
            Normalized tensor with matching input dtype and shape.

`FP32AccumulatorLayerNorm(normalized_shape: int | tuple[int, ...] | list[int] | torch.Size, eps: float = 1e-05, elementwise_affine: bool = True, bias: bool = True, device: torch.device | str | None = None, dtype: torch.dtype | None = None)`
:   Standard Layer Normalization with FP32 accumulator precision.
    
    Prevents FP16 numerical instability and variance underflow by computing
    mean and centered variance in `torch.float32`:
        $$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `extra_repr(self) ‑> str`
    :   Return the extra representation of the module.
        
        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Forward pass with FP32 accumulator computation.
        
        Args:
            x: Input tensor of shape `(..., *normalized_shape)`.
        
        Returns:
            Normalized tensor with matching input dtype and shape.

`RMSNorm(normalized_shape: int | tuple[int, ...] | list[int] | torch.Size, eps: float = 1e-06, elementwise_affine: bool = True, device: torch.device | str | None = None, dtype: torch.dtype | None = None)`
:   Root Mean Square Layer Normalization (RMSNorm) with FP32 accumulator precision.
    
    In standard FP16 mixed-precision training, squaring high-magnitude activations
    can exceed the FP16 maximum value (65504), causing overflow to `inf` / `nan`, or
    underflow to zero for small activations. This implementation explicitly casts
    the reduction accumulators to `torch.float32` before computing the sum-of-squares
    and reciprocal square root:
        $$y = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `extra_repr(self) ‑> str`
    :   Return the extra representation of the module.
        
        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Forward pass with FP32 accumulator computation.
        
        Args:
            x: Input tensor of shape `(..., *normalized_shape)`.
        
        Returns:
            Normalized tensor with matching input dtype and shape.

`SafeRMSNorm(normalized_shape: int | tuple[int, ...] | list[int] | torch.Size, eps: float = 1e-06, elementwise_affine: bool = True, device: torch.device | str | None = None, dtype: torch.dtype | None = None)`
:   Root Mean Square Layer Normalization (RMSNorm) with FP32 accumulator precision.
    
    In standard FP16 mixed-precision training, squaring high-magnitude activations
    can exceed the FP16 maximum value (65504), causing overflow to `inf` / `nan`, or
    underflow to zero for small activations. This implementation explicitly casts
    the reduction accumulators to `torch.float32` before computing the sum-of-squares
    and reciprocal square root:
        $$y = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `extra_repr(self) ‑> str`
    :   Return the extra representation of the module.
        
        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Forward pass with FP32 accumulator computation.
        
        Args:
            x: Input tensor of shape `(..., *normalized_shape)`.
        
        Returns:
            Normalized tensor with matching input dtype and shape.

`FP32AccumulatorRMSNorm(normalized_shape: int | tuple[int, ...] | list[int] | torch.Size, eps: float = 1e-06, elementwise_affine: bool = True, device: torch.device | str | None = None, dtype: torch.dtype | None = None)`
:   Root Mean Square Layer Normalization (RMSNorm) with FP32 accumulator precision.
    
    In standard FP16 mixed-precision training, squaring high-magnitude activations
    can exceed the FP16 maximum value (65504), causing overflow to `inf` / `nan`, or
    underflow to zero for small activations. This implementation explicitly casts
    the reduction accumulators to `torch.float32` before computing the sum-of-squares
    and reciprocal square root:
        $$y = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `extra_repr(self) ‑> str`
    :   Return the extra representation of the module.
        
        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Forward pass with FP32 accumulator computation.
        
        Args:
            x: Input tensor of shape `(..., *normalized_shape)`.
        
        Returns:
            Normalized tensor with matching input dtype and shape.