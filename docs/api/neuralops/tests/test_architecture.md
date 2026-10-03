Module neuralops.tests.test_architecture
========================================
Unit tests for core architecture discovery, leaf extraction, and attention injection.

Functions
---------

`set_random_seed()`
:   

Classes
-------

`TestGetAttentionModules()`
:   Verify get_attention_modules discovers only leaf submodules across all families.

    ### Methods

    `test_bert_returns_leaf_self_attention_only(self)`
    :

    `test_custom_nested_containers_deduplication(self)`
    :   Mock model with 3 levels of nested attention containers; assert only innermost leaf is returned.

    `test_distilbert_returns_leaf_attention(self)`
    :

    `test_gpt2_returns_correct_attention_block(self)`
    :

    `test_roberta_returns_leaf_self_attention_only(self)`
    :

    `test_t5_returns_encoder_and_decoder_attentions(self)`
    :

`TestInjectAttention()`
:   Verify inject_attention preserves residuals, LayerNorm, and forward semantics.

    ### Methods

    `test_gpt2_inject_conv1d_transposed_properly(self)`
    :

    `test_roberta_inject_preserves_output_dense_and_residual(self)`
    :

    `test_t5_encoder_inject(self)`
    :

`TestReplaceModuleDottedPath()`
:   Verify replace_module works correctly with nested dotted paths.

    ### Methods

    `test_nested_dotted_path_replacement(self)`
    :