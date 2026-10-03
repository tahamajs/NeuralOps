Module neuralops.tests.test_adasvd_hardened
===========================================

Functions
---------

`test_activation_whitening_and_weighted_als()`
:   Verify Cholesky whitening and weighted ALS reduce activation error.

`test_als_refinement_error_monotonicity()`
:   Verify that ALS refinement never increases error over initial SVD.

`test_cfg_get()`
:   Verify _cfg_get handles None, dict, and object attributes correctly.

`test_effective_rank_calculation()`
:   Verify Shannon entropy effective rank calculation.

`test_factory_with_none_and_dict_config()`
:   Verify adasvd_factory works with None, dict, and custom object configs.

`test_gating_aware_d_for_ffn_lin1()`
:   Test that ffn.lin1 gets gating-aware D, lin2 gets variance.

`test_gelu_derivative_squared()`
:   Test GELU derivative squared values at known points.

`test_lora_and_conv1d_detection_and_preservation()`
:   Verify LoRALinear wrappers are preserved and Conv1D weights transposed correctly.

`test_mean_shift_absorption()`
:   Verify mean-shift bias absorption centers activations and reduces reconstruction error.

`test_rank_floor_and_reduction_bounds()`
:   Verify minimum rank floor and parameter reduction constraints.

`test_reconstruction_error_and_fallback()`
:   Verify fallback to original layer when reconstruction error exceeds max_error.

`test_strs_rank_search()`
:   Verify STRS finds minimal rank satisfying max_error constraint.

`test_two_sided_als_improves_over_svd()`
:   Verify two-sided weighted ALS converges and provides competitive or improved error vs SVD.

Classes
-------

`DummyLoRAModel(in_f=64, out_f=64, r=4)`
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

    `forward(self, x) ‑> Callable[..., typing.Any]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.