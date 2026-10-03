Module neuralops.tests.test_svd_linear_attention
================================================

Functions
---------

`test_adapt_to_kv_size_modes()`
:   Test 9: GQA adaptation modes 'slice', 'mean', and 'first' work properly.

`test_chunked_decay_scan_bf16()`
:   Test 11: chunked decay scan handles BF16 tensors without type promotion crash.

`test_copy_weights_distilbert()`
:   Test 7: copy_weights successfully extracts weights from DistilBERT style attention.

`test_copy_weights_gpt2()`
:   Test 8: copy_weights successfully extracts weights from GPT-2 style fused c_attn and c_proj.

`test_copy_weights_near_zero_frobenius_raises()`
:   Test 6: copy_weights raises RuntimeError when projection weight has near-zero Frobenius norm.

`test_copy_weights_none_raises()`
:   Test 5: copy_weights raises RuntimeError when essential projections cannot be extracted.

`test_linear_kernel_and_recurrent()`
:   Test 12: linear kernel mode and step_recurrent execution.

`test_registry_and_injector()`
:   Test 13: ATTENTION_REGISTRY contains aliases and inject_svd_linear_attention works.

`test_sdpa_in_exact_mode()`
:   Test 10: exact mode uses SDPA when output_attentions=False, returns attn_probs when True.

`test_svdlinear_forward()`
:   Test 4: SVDLinear forward pass with and without RMSNorm.

`test_svdlinear_get_weight()`
:   Test 2: SVDLinear has get_weight() method and not weight property.

`test_svdlinear_no_custom_attention_attr()`
:   Test 1: SVDLinear does not have is_custom_attention = True as class attribute.

`test_svdlinear_set_low_rank_factors_zeroing()`
:   Test 3: set_low_rank_factors only zeroes written slice and handles smaller rank.