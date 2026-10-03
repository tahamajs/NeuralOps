Module neuralops.tests.test_triton_cosformer_comprehensive
==========================================================

Functions
---------

`test_triton_cosformer_attention_masking()`
:   Verify attention_mask zeros out padded tokens and prevents gradient contamination.

`test_triton_cosformer_bidirectional_and_causal_shapes()`
:   Verify forward passes in both causal (decoder) and bidirectional (encoder) modes.

`test_triton_cosformer_clean_autograd()`
:   Verify backward executes with torch.autograd.grad without double-backward warnings.

`test_triton_cosformer_config_handling()`
:   Verify safe config handling when config is None, SimpleNamespace, or full Config.

`test_triton_cosformer_injection_pipeline()`
:   Verify inject_triton_cosformer installs cleanly into a model with attention layers.

`test_triton_cosformer_rope_support()`
:   Verify RoPE rotary position embeddings are applied to Q and K when provided.

`test_triton_cosformer_weight_copy_multi_arch()`
:   Verify copy_weights correctly extracts projections from GPT-2, RoBERTa, and DistilBERT.