Module neuralops.tests.test_hybrid_attention
============================================

Functions
---------

`test_alias_class()`
:   WeightedEnsembleAttention is an alias for HybridAttention.

`test_basic_construction()`
:   Basic construction with minimal args.

`test_causal_future_does_not_affect_past()`
:   In causal mode, changing future tokens does not affect past.

`test_causal_mask_no_nan()`
:   Causal mode produces valid output.

`test_forward_shape()`
:   Forward produces correct output shape.

`test_forward_with_mask()`
:   Forward with attention mask works.

`test_gate_init_is_softmax_dominant()`
:   At init, gate ≈ 1 so model ≈ standard attention.

`test_long_sequence_memory_bounded()`
:   Chunked scan keeps memory sub-quadratic.

`test_nested_encoder_detection()`
:   BertSelfAttention-like class is detected as nested.

`test_no_layernorm_added()`
:   HybridAttention should not add internal LayerNorm.

`test_parameter_count()`
:   Parameter count is within expected range.

`test_registry_keys_preserved()`
:   Old registry keys still work.

`test_valid_mask_extraction()`
:   _extract_valid_mask handles all dims and dtypes.