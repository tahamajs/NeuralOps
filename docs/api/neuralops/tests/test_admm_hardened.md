Module neuralops.tests.test_admm_hardened
=========================================
Unit tests for production-hardened ADMM Compressor.

Verifies:
1. Support for LoRALinear and Conv1D (shape transposition, preserving LoRA wrappers, trainable LoRA).
2. Factory robustness when comp_config is None, dict, or object.
3. Reconstruction error validation and fallback mechanism when error exceeds max_error.
4. Quantized + pruned weight shape matches original and contains no NaN / Inf.
5. target_attention_only=True compresses attention layers while strictly skipping FFN/MLP.
6. Fallback layers are strictly frozen (requires_grad=False) during recovery parameter selection.

Classes
-------

`MockConv1D(in_features: int, out_features: int)`
:   Mock of HuggingFace Conv1D (GPT-2 style) with weight shape [in_features, out_features].
    
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

`MockLoRALinear(base_layer: torch.nn.modules.module.Module, r: int = 16)`
:   Mock of LoRALinear wrapper around a base layer.
    
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

`TestADMMHardened(methodName='runTest')`
:   A class whose instances are single test cases.
    
    By default, the test code itself should be placed in a method named
    'runTest'.
    
    If the fixture may be used for many test cases, create as
    many test methods as are needed. When instantiating such a TestCase
    subclass, specify in the constructor arguments the name of the test method
    that the instance is to execute.
    
    Test authors should subclass TestCase for their own tests. Construction
    and deconstruction of the test's environment ('fixture') can be
    implemented by overriding the 'setUp' and 'tearDown' methods respectively.
    
    If it is necessary to override the __init__ method, the base class
    __init__ method must always be called. It is important that subclasses
    should not change the signature of their __init__ method, since instances
    of the classes are instantiated automatically by parts of the framework
    in order to be run.
    
    When subclassing TestCase, you can set these attributes:
    * failureException: determines which exception will be raised when
        the instance's assertion methods fail; test methods raising this
        exception will be deemed to have 'failed' rather than 'errored'.
    * longMessage: determines whether long messages (including repr of
        objects used in assert methods) will be printed on failure in *addition*
        to any explicit message passed.
    * maxDiff: sets the maximum length of a diff in failure messages
        by assert methods using difflib. It is looked up as an instance
        attribute so can be configured by individual tests if required.
    
    Create an instance of the class that will use the named test
    method when executed. Raises a ValueError if the instance does
    not have a method with the specified name.

    ### Ancestors (in MRO)

    * unittest.case.TestCase

    ### Methods

    `setUp(self)`
    :   Hook method for setting up the test fixture before exercising it.

    `test_admm_factory_with_none_config(self)`
    :   2. Verify admm_factory works safely with None, dict, and config object.

    `test_fallback_layers_are_frozen(self)`
    :   6. Verify that fallback layers have requires_grad=False and are excluded from recovery.

    `test_lora_linear_and_conv1d_detection(self)`
    :   1. Verify LoRALinear and Conv1D detection, shape transposition, and LoRA preservation.

    `test_quantized_pruned_weight_properties(self)`
    :   4. Verify weight shape match, no NaN/Inf, and 2:4 structured sparsity pattern.

    `test_reconstruction_error_and_fallback(self)`
    :   3. Verify error thresholding triggers fallback to original weights when error > max_error.

    `test_target_attention_only_skips_ffn(self)`
    :   5. Verify target_attention_only=True strictly skips MLP/FFN and classifier.

`ToyTransformer()`
:   Toy model containing attention and FFN layers for filtering tests.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module