Module neuralops.tests.test_adaptive_span_comprehensive
=======================================================

Functions
---------

`test_adaptive_span_bidirectional_and_causal_shapes()`
:   

`test_adaptive_span_cross_attention_bypass()`
:   3. Verify cross-attention with unequal N_q and N_k safely bypasses span masking without crashing.

`test_adaptive_span_differentiability()`
:   Verify that span_logits is fully differentiable and receives non-zero gradients.

`test_adaptive_span_fallback_on_copy_weights_failure()`
:   

`test_adaptive_span_frobenius_norm_verification()`
:   

`test_adaptive_span_gqa_asymmetry()`
:   2. Verify GQA repeat_kv logic expands tensors without dimension mismatches when num_heads > num_kv_heads.

`test_adaptive_span_injection_all_4_architectures()`
:   

`test_adaptive_span_lora_adapters_freezing()`
:   

`test_adaptive_span_lora_config_independence()`
:   

`test_adaptive_span_masking()`
:   

`test_adaptive_span_nested_encoder_bypass()`
:   4. Verify nested encoder bypass (RoBERTa/BERT): q_proj gets grads, out_proj has no trainable LoRA grads.

`test_adaptive_span_nested_encoder_outer_dense_wrap()`
:   Verify inject_adaptive_span wraps outer attention.output.dense with LoRALinear for RoBERTa/BERT.

`test_adaptive_span_posterior_renormalization()`
:   1. Verify attention weights sum to exactly 1.0 along dim=-1 after soft ramp and re-normalization.

`test_adaptive_span_rotary_fallback()`
:   

`test_adaptive_span_zero_span_stability()`
:   5. Verify numerical stability when span is ~0 (span_logits = -100.0) with zero NaNs or Infs.

`test_lora_linear_canonical_contract()`
:   Verify LoRALinear adheres strictly to canonical LoRA without extraneous output_scale.