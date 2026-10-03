Module neuralops.tests.test_full_framework_end_to_end
=====================================================
Master End-to-End Test Suite for All Framework Features and Subsystems.

Tests:
1. Core Configuration & Serialization
2. Registries (Attentions & Compressors)
3. Universal Weight Transfer (Adapters)
4. Attention Mask Subsystem (2D, 3D, 4D, Causal, Sliding Window)
5. Attention Injection Pipeline
6. Compression Engine Lifecycle
7. Evaluation Metrics & Profiler
8. Complete Trainer Optimization Loop
9. Worker Status & Telemetry Output

Classes
-------

`TestFullFrameworkEndToEnd(methodName='runTest')`
:   Comprehensive test case covering all framework components and execution paths.
    
    Create an instance of the class that will use the named test
    method when executed. Raises a ValueError if the instance does
    not have a method with the specified name.

    ### Ancestors (in MRO)

    * unittest.case.TestCase

    ### Methods

    `setUp(self) ‑> None`
    :   Hook method for setting up the test fixture before exercising it.

    `tearDown(self) ‑> None`
    :   Hook method for deconstructing the test fixture after testing it.

    `test_01_configuration_lifecycle(self) ‑> None`
    :   Test ExperimentConfig creation and serialization.

    `test_02_registries_completeness(self) ‑> None`
    :   Verify ATTENTION_REGISTRY and COMPRESSOR_REGISTRY contain all expected modules.

    `test_03_universal_adapter_weight_extraction(self) ‑> None`
    :   Verify UniversalAdapter extracts Q, K, V, and Output projections from diverse blocks.

    `test_04_attention_mask_subsystem(self) ‑> None`
    :   Verify attention mask helpers across 2D, 3D, 4D, and causal configurations.

    `test_05_attention_injection_pipeline(self) ‑> None`
    :   Verify dynamic injection of custom attention modules into transformer backbones.

    `test_06_compression_engine_lifecycle(self) ‑> None`
    :   Verify compression engine factorizes linear modules and returns accurate statistics.

    `test_07_metrics_and_profiler(self) ‑> None`
    :   Verify accuracy/f1 computation and ModelProfiler forward hooks.

    `test_08_trainer_execution_loop(self) ‑> None`
    :   Verify end-to-end training and evaluation loops on synthetic batches.

    `test_09_worker_telemetry_serialization(self) ‑> None`
    :   Verify worker telemetry status file creation and JSON schema.

`ToyClassifierModel(hidden_size: int = 64, num_layers: int = 2, num_classes: int = 2)`
:   Mock sequence classification model with transformer layers.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) ‑> Any`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`ToyEncoderBlock(hidden_size: int = 64, num_heads: int = 4)`
:   Mock transformer block matching DistilBERT architecture.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.