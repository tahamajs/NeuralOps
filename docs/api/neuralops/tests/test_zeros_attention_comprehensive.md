Module neuralops.tests.test_zeros_attention_comprehensive
=========================================================

Functions
---------

`make_config(hidden_size=64, num_heads=4, is_decoder=False, model_type='bert')`
:   

`test_bidirectional_shape_and_centering()`
:   

`test_causal_shape_and_causality()`
:   

`test_four_architectures_forward()`
:   

`test_head_mask_broadcasting()`
:   

`test_inject_with_none_old_attn_returns_none()`
:   

`test_lora_adapter_preservation()`
:   

`test_rope_and_even_odd_dim_safety()`
:   

Classes
-------

`DummyAttn(d=64)`
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