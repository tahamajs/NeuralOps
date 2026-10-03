Module neuralops.tests.test_integration_smoke
=============================================
Integration smoke tests verifying end-to-end injection and forward passes per model family.

Functions
---------

`set_seed()`
:   

Classes
-------

`TestIntegrationSmokePerFamily()`
:   Verify end-to-end forward and backward passes across model families.

    ### Methods

    `test_bert_rpc_attention_smoke(self)`
    :

    `test_distilbert_adaptive_svd_smoke(self)`
    :

    `test_gpt2_adaptive_svd_smoke(self)`
    :

    `test_incompatible_causal_skip_smoke(self)`
    :   Incompatible causal module is cleanly identified for RoBERTa without crashing model.

    `test_roberta_adaptive_svd_smoke(self)`
    :

    `test_roberta_compressor_smoke(self)`
    :

    `test_t5_encoder_adaptive_svd_smoke(self)`
    :