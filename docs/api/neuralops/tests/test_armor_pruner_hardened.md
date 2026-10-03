Module neuralops.tests.test_armor_pruner_hardened
=================================================
Comprehensive Unit Test Suite for Production-Hardened ARMOR Pruner.
Covers:
1. Factory config safety (None, dict, missing parameters)
2. Conv1D transposition & shape compatibility
3. LoRALinear wrapper preservation and adapter policy
4. Fallback behavior (max_error thresholding & fallback_to_original toggle)
5. Target attention only filtering (attention vs FFN / nested projections)
6. Multi-batch calibration DataLoader iteration
7. Numerical stability in FP16 / FP32 (no NaN / Inf)
8. Cross-architecture compatibility (DistilBERT, RoBERTa, GPT-2, Qwen3.5 dummy backbones)

Functions
---------

`test_architecture_compatibility()`
:   Test 8: Compatibility across dummy models representing diverse transformer families.

`test_calibration_with_dataloader()`
:   Test 6: Calibration loader iterates over all batches and aggregates activations.

`test_conv1d_transposition()`
:   Test 2: Conv1D [in_features, out_features] is correctly transposed and replaced by ARMORLinear.

`test_factory_config_safety()`
:   Test 1: Factory with None, empty, or dict comp_config does not crash and respects overrides.

`test_fallback_behavior()`
:   Test 4: Fallback triggered when error > max_error with parameter freezing.

`test_loralinear_wrapper_preservation()`
:   Test 3: Wrapper is preserved, base_layer is replaced, wrapper LoRA zeroed, ARMOR LoRA trainable.

`test_numerical_stability()`
:   Test 7: FP16 tensors do not produce NaN/Inf in parameters or forward outputs.

`test_target_attention_only()`
:   Test 5: target_attention_only=True compresses only attention projections, strictly skipping FFN.

Classes
-------

`DummyAttentionFFNModel(d: int = 16)`
:   Model with both attention projections and FFN / MLP projections.
    
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

`DummyConv1D(in_features: int, out_features: int)`
:   Mimics transformers.pytorch_utils.Conv1D (weights are [in_features, out_features]).
    
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

`DummyLoRALinear(base_layer: torch.nn.modules.module.Module, r: int = 4)`
:   Mimics LoRALinear wrapper with base_layer and adapter branches.
    
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