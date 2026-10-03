Module neuralops.tests.test_upi_mamba_comprehensive
===================================================

Functions
---------

`make_config(hidden_size=64, is_decoder=False, model_type='bert')`
:   

`test_upi_mamba_causal_mode()`
:   

`test_upi_mamba_copy_weights_all_4_architectures()`
:   

`test_upi_mamba_forward_and_shapes()`
:   

`test_upi_mamba_inject()`
:   

`test_upi_mamba_is_selective()`
:   Verify the SSM is truly selective: B_t, C_t depend on input, and a spike evolves state.

`test_upi_mamba_lora_and_gradients()`
:   

`test_upi_mamba_masks()`
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