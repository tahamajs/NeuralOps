Module neuralops.tests.test_salt_f
==================================

Functions
---------

`test_attention_aware_gamma_init()`
:   Gamma init from sensitivity changes the values.

`test_fisher_weighted_svd()`
:   Fisher weighting changes the SVD.

`test_merge_model()`
:   merge_model_salt replaces SALTLinear with nn.Linear.

`test_merge_to_linear()`
:   Merged linear produces the same output.

`test_per_projection_adaptive_rank()`
:   Different projections get different ranks.

`test_registry_registration()`
:   SALT keys are registered.

`test_salt_linear_construction()`
:   Basic SALTLinear construction.

`test_salt_linear_init_recovers_W_r()`
:   Init should give W_r (truncated SVD).