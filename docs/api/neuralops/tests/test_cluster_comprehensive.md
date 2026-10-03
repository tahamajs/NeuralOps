Module neuralops.tests.test_cluster_comprehensive
=================================================

Functions
---------

`test_cluster_bidirectional_and_causal_shapes()`
:   

`test_cluster_config_none_no_attr_error()`
:   config=None must not raise AttributeError when checking is_decoder in forward.

`test_cluster_fallback_on_copy_weights_failure()`
:   

`test_cluster_frobenius_norm_verification()`
:   

`test_cluster_head_dim_fallback()`
:   

`test_cluster_injection_all_4_architectures()`
:   

`test_cluster_kmeans_forward()`
:   

`test_cluster_lora_adapters_freezing()`
:   

`test_cluster_lora_config_independence()`
:   

`test_cluster_num_kv_heads_from_source_attn()`
:   num_kv_heads should be picked up from source_attn when config=None.

`test_cluster_padding_mask_query_zeroing()`
:   

`test_cluster_rope_uses_query_not_key()`
:   RoPE fallback must call rotary_emb with Q (queries), not K.

`test_cluster_rotary_fallback()`
:   

`test_cluster_third_rope_fallback_signature()`
: