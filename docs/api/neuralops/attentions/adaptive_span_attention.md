Module neuralops.attentions.adaptive_span_attention
===================================================
Adaptive Span Attention — Final Corrected Implementation.
Based on: Sukhbaatar et al., 2019 (arXiv:1905.07799)

Math:
    z_h = sigmoid(logit_h) * S_max
    g_{z_h}(d) = clamp((R + z_h - d) / R, 0, 1)

    # FIX applied: The mask must be applied INSIDE the softmax
    A = Softmax(S + log(g_z) + M)
    O = (A V) W_O

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Apply RoPE using absolute position_ids.

`get_total_span_loss(model: nn.Module, lambda_span: float = 0.01) ‑> torch.Tensor`
:   Total L1 span regularization loss across all AdaptiveSpanAttention modules.
    
    L_span = lambda * (1/|M|) * sum_m mean_h(sigmoid(logit_mh))

`inject_adaptive_span(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any) ‑> neuralops.attentions.adaptive_span_attention.AdaptiveSpanAttention | torch.nn.modules.module.Module | None`
:   

Classes
-------

`AdaptiveSpanAttention(config: Any | None = None, rank: int | None = None, dropout: float = 0.1, max_span: int = 128, ramp_size: float = 32.0, lora_rank: int = 16, comp_config: Any | None = None, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, layer_idx: int | None = None, return_style: str | None = None, rolling_cache_size: int | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Instance variables

    `current_span: torch.Tensor`
    :

    ### Methods

    `compute_bias(self, query_length: int, key_length: int, device=None) ‑> torch.Tensor`
    :

    `get_span_regularization_loss(self, lambda_span: float = 0.01) ‑> torch.Tensor`
    :

`IdentityOutProj()`
:   A placeholder identity operator that is argument-insensitive.
    
    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)
    
    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    
    Examples::
    
        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.linear.Identity
    * torch.nn.modules.module.Module

    ### Methods

    `freeze_base_weights(self) ‑> None`
    :

`LoRALinear(in_features: int, out_features: int, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `freeze_base_weights(self) ‑> None`
    :

    `reset_parameters(self) ‑> None`
    :