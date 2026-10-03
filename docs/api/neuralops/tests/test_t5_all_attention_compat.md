Module neuralops.tests.test_t5_all_attention_compat
===================================================
Unit tests for all 27 attention replacement modules adapted for T5.
Verifies:
1. Forward pass is finite (no NaN/Inf).
2. Shapes are correct [B, num_classes].
3. Relative position bias is applied and propagated.
4. Cross-attention uses different sources for Q (decoder) vs K/V (encoder).
5. Encoder self-attention, decoder self-attention, and decoder cross-attention all work.
6. FP16 numerical stability.

Functions
---------

`t5_base_model()`
:   

`test_cross_attention_different_sources(t5_base_model, module_name)`
:   Verify that decoder cross-attention uses encoder states for K/V and decoder states for Q.

`test_forward_pass_finite_and_shape(t5_base_model, module_name)`
:   Verify forward pass is finite and produces correct output shape [B, 2].

`test_fp16_numerical_stability(t5_base_model, module_name)`
:   Verify that running in FP16 produces no NaNs.

`test_relative_position_bias_applied(t5_base_model, module_name)`
:   Verify that relative position bias is attached and used in layer 0.