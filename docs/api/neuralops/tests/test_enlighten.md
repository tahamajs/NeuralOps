Module neuralops.tests.test_enlighten
=====================================
Comprehensive test suite for ENLighten compressor.

Tests:
1. Registration under both 'enlighten' and 'enlighten_compressor'.
2. Vectorized 2:4 structured sparsity patterns across row blocks.
3. Unstructured magnitude pruning fraction accuracy.
4. Linear layer compression and error calculation without artificial scale distortion.
5. Conv1D / transposed layer support (GPT-2 style).
6. Target attention only filtering and non-target freezing.
7. Iterative rank escalation on high error.
8. Safe fallback to original module if error threshold cannot be met.
9. LoRA adapter preservation on wrapped modules.
10. Factory function invocation and registry lookup.

Functions
---------

`test_enlighten_compression_forward()`
:   Verify model compression and clean forward pass.

`test_factory_function()`
:   Verify enlighten_factory works with registry call.

`test_fallback_on_unreachable_error()`
:   Verify fallback to original module if max_error threshold cannot be met.

`test_no_scale_distortion()`
:   Verify that _decompose_and_prune does not multiply by an arbitrary Frobenius norm scale factor.

`test_registry()`
:   Verify registration under both canonical names.

`test_target_attention_only()`
:   Verify target_attention_only restricts compression to attention projections.

`test_unstructured_prune()`
:   Verify unstructured magnitude pruning achieves target sparsity.

`test_vectorized_24_pruning()`
:   Verify exactly 2 out of every 4 entries are non-zero in each block.

Classes
-------

`DummyLinearModel(in_dim: int = 64, out_dim: int = 64)`
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