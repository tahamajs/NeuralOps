Module neuralops.tests.test_attention_modules_compat
====================================================
Unit tests for attention module family compatibility and declared capabilities.

Classes
-------

`DummyConfig(hidden_size=32, num_attention_heads=2, num_heads=2, **kwargs)`
:   DummyConfig(hidden_size=32, num_attention_heads=2, num_heads=2, **kwargs)

    ### Ancestors (in MRO)

    * transformers.configuration_utils.PreTrainedConfig
    * transformers.utils.hub.PushToHubMixin
    * transformers.modeling_rope_utils.RotaryEmbeddingConfigMixin

`TestAttentionFamilyCompatibility()`
:   Verify capability declarations and family compatibility matrices.

    ### Methods

    `test_bidirectional_compatible_with_both(self, module_name)`
    :   Bidirectional attention modules must be compatible with both encoders and decoders.

    `test_causal_only_compatible_with_decoders(self, module_name)`
    :   Causal-only modules must be allowed for GPT-2, Qwen, LLaMA.

    `test_causal_only_incompatible_with_encoders(self, module_name)`
    :   Causal-only modules must be rejected for BERT, RoBERTa, DistilBERT.

    `test_instantiate_representative_modules(self, mod_name)`
    :   Verify representative registered modules instantiate with a minimal dummy config.

    `test_module_family_support_dictionary(self)`
    :   Verify module_family_support returns structured dictionary across families.