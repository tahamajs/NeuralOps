Module neuralops.tests.test_wgqa_comprehensive
==============================================

Functions
---------

`make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=False, model_type='bert')`
:   

`test_causal_shape_and_causality()`
:   

`test_gqa_pooling()`
:   

`test_head_mask_broadcasting()`
:   

`test_inject_wgqa()`
:   

`test_lora_adapters_and_freezing()`
:   

`test_rope_and_even_odd_dim_safety()`
:   

`test_wgqa_forward_and_head_weights()`
:   

Classes
-------

`DummyAttn(d=64, h=4)`
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