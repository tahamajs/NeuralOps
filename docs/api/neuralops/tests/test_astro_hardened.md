Module neuralops.tests.test_astro_hardened
==========================================
Unit tests for hardened AstroQuantizer with Fisher-Adaptive Bit Allocation.
Verifies:
1. allocate_fisher_bits solves the integer knapsack budget and averages to target_avg_bits.
2. quantize_dequantize_tensor with heterogeneous per-channel bits produces valid finite values and shapes.
3. LoRA wrapper and Conv1D detection and preservation.
4. target_attention_only=True skips FFN/MLP layers and retains attention projections.
5. High-error fallback triggers, restores weights, and freezes fallback parameters (requires_grad=False).
6. Factory works with comp_config=None and records effective_compression_rate and theoretical_vram_saving.
7. STE autograd backpropagation across heterogeneous bits.

Classes
-------

`DummyLoRALinear(base_layer: nn.Module, rank: int = 4)`
:   Mock LoRA linear layer wrapping base_layer.
    
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

`DummyTransformerModel()`
:   Mock transformer block with attention and FFN layers for testing.
    
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

`TestAstroHardened(methodName='runTest')`
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

    `test_01_allocate_fisher_bits_budget(self)`
    :   Verify allocate_fisher_bits matches target_avg_bits within 1/N tolerance.

    `test_02_per_channel_quantization_shapes_and_values(self)`
    :   Verify heterogeneous per-channel quantization produces correct shapes and finite numbers.

    `test_03_ste_autograd_flow(self)`
    :   Verify STE backpropagation flows gradients across heterogeneous per-channel bit allocations.

    `test_04_target_attention_only_skips_ffn(self)`
    :   Verify target_attention_only=True only compresses attention layers and skips FFN and classifier.

    `test_05_lora_conv1d_support_and_trainability(self)`
    :   Verify LoRALinear base_layer is quantized while LoRA adapters remain trainable.

    `test_06_fallback_and_freezing(self)`
    :   Verify layer exceeding max_error falls back to original and has its parameters frozen.

    `test_07_factory_with_none_config_and_metrics(self)`
    :   Verify astro_factory works with comp_config=None and returns valid metrics.