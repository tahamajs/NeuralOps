Module neuralops.tests.test_adasplash_attention
===============================================
Hardening tests for AdaSplashAttention (adasplash_attention.py).

Covers: fail-loud copy_weights, odd-dim RoPE, inject safety, forward shapes.

Classes
-------

`DummyDistilBertAttn()`
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

`DummyEmptyAttn(*args: Any, **kwargs: Any)`
:   Attention with no extractable projections — should trigger RuntimeError.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x) ‑> Callable[..., typing.Any]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`DummyGPT2Attn()`
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

`DummyLayer(attn)`
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

`DummyQwenAttn()`
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

`DummyRobertaAttn()`
:   Simulates RoBERTa BertAttention with nested .self / .output.dense.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

`MockConfig()`
:   

    ### Class variables

    `alpha`
    :   The type of the None singleton.

    `head_dim`
    :   The type of the None singleton.

    `hidden_size`
    :   The type of the None singleton.

    `is_decoder`
    :   The type of the None singleton.

    `lora_rank`
    :   The type of the None singleton.

    `model_type`
    :   The type of the None singleton.

    `num_attention_heads`
    :   The type of the None singleton.

    `num_heads`
    :   The type of the None singleton.

    `num_key_value_heads`
    :   The type of the None singleton.

    `rank`
    :   The type of the None singleton.

`TestCopyWeightsFailLoud()`
:   

    ### Methods

    `test_copy_distilbert_succeeds(self)`
    :

    `test_copy_empty_not_warns_silently(self)`
    :   copy_weights must NOT silently return — it must raise.

    `test_copy_empty_raises_runtimeerror(self)`
    :

    `test_copy_gpt2_succeeds(self)`
    :

    `test_copy_qwen_succeeds(self)`
    :

    `test_copy_roberta_succeeds(self)`
    :

`TestForwardShapes()`
:   

    ### Methods

    `test_encoder_output_shape(self)`
    :

    `test_output_attentions(self)`
    :

    `test_pretrained_attn_in_constructor(self)`
    :

    `test_with_attention_mask(self)`
    :

`TestInjectSafety()`
:   

    ### Methods

    `test_inject_distilbert(self)`
    :

    `test_inject_empty_attn_returns_none(self)`
    :   Inject with an unrecognised attn (no projections) must return None, not raise.

    `test_inject_none_returns_none(self)`
    :

    `test_inject_qwen(self)`
    :

    `test_inject_returns_adasplash_instance(self)`
    :

`TestOddDimRoPE()`
:   

    ### Methods

    `test_forward_even_dim_no_rope(self)`
    :

    `test_forward_with_rope_even(self)`
    :

    `test_forward_with_rope_odd_head_dim(self)`
    :   Odd head_dim must not crash (e.g. head_dim=7).