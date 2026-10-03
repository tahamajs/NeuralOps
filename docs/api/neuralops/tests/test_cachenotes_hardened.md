Module neuralops.tests.test_cachenotes_hardened
===============================================
Unit tests for hardened CacheNotes compressor.
Verifies:
1. LoRA wrapper and Conv1D detection and transposition.
2. If layer is already LoRALinear, base_layer is compressed without creating redundant wrappers, and LoRA adapters remain trainable.
3. CacheNotesLoRALinear forward matches x @ weight.T + bias + lora_out.
4. target_attention_only=True skips FFN layers and preserves classifier head.
5. High reconstruction error triggers fallback and freezes fallback parameters (requires_grad=False).
6. CacheNotesKVCache prunes sequence length while strictly preserving sink tokens.
7. Factory works with comp_config=None and returns compression_stats.

Classes
-------

`DummyLoRALinear(base_layer: nn.Module, rank: int = 4)`
:   Mock pre-existing LoRA linear layer wrapping base_layer.
    
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

`TestCacheNotesHardened(methodName='runTest')`
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

    `test_01_cachenotes_lora_linear_forward_exactness(self)`
    :   Verify CacheNotesLoRALinear forward matches x @ weight.T + bias + lora_out.

    `test_02_lora_and_conv1d_support(self)`
    :   Verify existing LoRALinear base_layer is compressed without dropping adapters, and Conv1D works.

    `test_03_target_attention_only_skips_ffn(self)`
    :   Verify target_attention_only=True skips FFN and classifier.

    `test_04_fallback_and_freezing(self)`
    :   Verify layer exceeding max_error falls back and has its parameters frozen.

    `test_05_kv_cache_pruning_and_sink_preservation(self)`
    :   Verify CacheNotesKVCache prunes tokens while strictly preserving initial sink tokens.

    `test_06_factory_with_none_config_and_metrics(self)`
    :   Verify cachenotes_factory works with comp_config=None and records correct metrics.

    `test_07_cpt_generator_multiple_dimensions_and_training(self)`
    :   Verify learnable CPTGenerator produces valid vectors across arbitrary dimensions and trains properly.