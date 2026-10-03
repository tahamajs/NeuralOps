Module neuralops.tests.test_attention_injection
===============================================
Tests for the attention injection utilities.

Classes
-------

`TestAttentionInjection()`
:   Verify that inject_attention works correctly on foundation transformer models.

    ### Methods

    `test_available_attention_types(self)`
    :   Verify registry queries return non-empty list of attention variants.

    `test_inject_attention_on_distilbert(self)`
    :   Inject known attention types into DistilBERT and verify forward pass execution.

    `test_inject_attention_with_copy_weights(self)`
    :   Verify injection with copy_weights enabled.