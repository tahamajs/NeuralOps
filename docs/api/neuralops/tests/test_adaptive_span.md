Module neuralops.tests.test_adaptive_span
=========================================
Unit tests for AdaptiveSpanAttention module.
Verifies all 8 core properties:
1. Module instantiates with correct span initialization (span_logits = 4.0, sigmoid ~ 0.982).
2. Span regularization loss is non-negative and strictly positive for initialized spans.
3. Multi-module span loss aggregation: get_total_span_loss computes correct average across layers.
4. Forward pass produces expected shape [B, N, D] and finite outputs.
5. Softmax NaN safety: completely masked rows produce 0 attention without NaN or inf.
6. Gradient flow: span_logits receives non-zero gradients when span regularization loss is backpropagated.
7. Registry registration: 'adaptive_span' and 'adaptive_span_attention' are registered.
8. Supports causal, bidirectional, and GQA flags appropriately.

Functions
---------

`test_1_instantiation_and_span_init()`
:   Verify AdaptiveSpanAttention initializes with span_logits=4.0 and current_span near max_span.

`test_2_span_regularization_loss_positive()`
:   Verify get_span_regularization_loss returns a positive scalar loss.

`test_3_multi_module_span_loss_aggregation()`
:   Verify get_total_span_loss aggregates across multiple layers in a model.

`test_4_forward_pass_output_shape()`
:   Verify forward pass returns expected output shape [B, N, D] with finite values.

`test_5_softmax_nan_safety_all_masked()`
:   Verify fully masked rows do not result in NaNs in attention or output.

`test_6_gradient_flow_to_span_logits()`
:   Verify span_logits receives non-zero gradients when span regularization loss is backpropagated.

`test_7_registry_keys()`
:   Verify 'adaptive_span' and 'adaptive_span_attention' are present in ATTENTION_REGISTRY.

`test_8_attention_flags()`
:   Verify supports_causal, supports_bidirectional, supports_gqa, and is_custom_attention flags.