Module neuralops.tests.test_weight_copy
=======================================
Unit tests for weight copying, QKV slicing, and projection detection.

Functions
---------

`set_random_seed()`
:   

Classes
-------

`TestCopyProjectionWeights()`
:   Verify _copy_projection_weights across all module conventions and safeguards.

    ### Methods

    `test_graceful_missing_projection_fallback(self)`
    :   Target module with unexpected structure logs warning and does not crash.

    `test_property_target_protection(self)`
    :   Verify that layers with @property weight are safely handled without crash.

    `test_standard_linear_copy(self, proj_name)`
    :

`TestSliceFusedQKVWeights()`
:   Verify _slice_fused_qkv_weights handles Conv1D and Linear fused QKV projections.

    ### Methods

    `test_conv1d_transposition_and_slicing(self)`
    :

    `test_missing_bias_handling(self)`
    :

    `test_standard_linear_fused_slicing(self)`
    :

`TestUniversalAdapterAndModelAdapter()`
:   Verify get_weight_bias retrieves weights correctly across naming conventions.

    ### Methods

    `test_canonical_aliases_coverage(self)`
    :

    `test_universal_adapter_linear(self)`
    :