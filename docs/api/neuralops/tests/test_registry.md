Module neuralops.tests.test_registry
====================================
Unit tests for central registry, module registration, and helper module filtering.

Classes
-------

`DummyConfig(hidden_size=32, num_attention_heads=2, **kwargs)`
:   DummyConfig(hidden_size=32, num_attention_heads=2, **kwargs)

    ### Ancestors (in MRO)

    * transformers.configuration_utils.PreTrainedConfig
    * transformers.utils.hub.PushToHubMixin
    * transformers.modeling_rope_utils.RotaryEmbeddingConfigMixin

`TestRegistryCore()`
:   Verify Registry mechanics, duplicate registration handling, and helper exclusion.

    ### Methods

    `test_core_registries_populated(self)`
    :

    `test_duplicate_registration_aliases(self)`
    :   Registering same class under multiple alias keys must work cleanly without error.

    `test_get_evaluatable_modules_non_empty(self)`
    :

    `test_is_helper_module_filtering(self)`
    :   Helper suffix patterns must be filtered out of evaluatable components.

    `test_registry_registration_and_retrieval(self)`
    :