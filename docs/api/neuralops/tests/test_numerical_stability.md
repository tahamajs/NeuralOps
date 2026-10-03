Module neuralops.tests.test_numerical_stability
===============================================
Unit tests for numerical stability, FP16/FP32 precision, and mask edge cases.

Functions
---------

`set_seed()`
:   

Classes
-------

`DummyConfig(hidden_size=32, num_attention_heads=2, **kwargs)`
:   DummyConfig(hidden_size=32, num_attention_heads=2, **kwargs)

    ### Ancestors (in MRO)

    * transformers.configuration_utils.PreTrainedConfig
    * transformers.utils.hub.PushToHubMixin
    * transformers.modeling_rope_utils.RotaryEmbeddingConfigMixin

`TestNumericalStability()`
:   Verify numerical finite outputs and gradients under fp32 and fp16.

    ### Methods

    `test_attention_mask_padding_and_large_negatives(self)`
    :   Verify 2D boolean mask and 4D additive large negative mask (-1e4).

    `test_fp32_forward_backward_finite(self, mod_name)`
    :

    `test_gradscaler_double_unscale_safe(self)`
    :   Verify that calling unscale_ twice does not unhandled-crash when guarded.

    `test_head_mask_support(self)`
    :   Verify wrapper cleanly processes head_mask without crashing.

    `test_performer_fp16_exp_clamping_prevents_overflow(self)`
    :   Verify that large inputs into Performer do not cause exp overflow to Inf under fp16.

    `test_sequence_length_512_stability(self)`
    :   Verify attention forward stability on longer sequence length 512.