Module neuralops.tests.test_tpa_comprehensive
=============================================

Functions
---------

`make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=False, model_type='bert')`
:   

`test_tpa_causal_mode()`
:   

`test_tpa_forward_and_shapes()`
:   

`test_tpa_gpt2_is_decoder_detection()`
:   

`test_tpa_gradient_checkpointing_toggle()`
:   

`test_tpa_inject()`
:   

`test_tpa_lora_and_gradients()`
:   

`test_tpa_masks()`
:   

`test_tpa_nested_encoder_injection()`
:   

`test_tpa_no_output_scale()`
:   

`test_tpa_non_square_head_dim()`
:   

`test_tpa_orthogonal_init()`
:   

`test_tpa_qwen_head_dim_and_rank()`
:   

`test_tpa_rotate_half_and_odd_dim()`
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