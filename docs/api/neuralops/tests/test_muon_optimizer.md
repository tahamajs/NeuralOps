Module neuralops.tests.test_muon_optimizer
==========================================

Functions
---------

`test_create_muon_adamw_optimizer_split()`
:   Verify create_muon_adamw_optimizer cleanly partitions 2D vs 1D parameters.

`test_muon_adamw_training_loop()`
:   Verify end-to-end training loop with MuonAdamW hybrid optimizer.

`test_muon_parameter_validation()`
:   Verify Muon strictly enforces 2D matrix parameters.

`test_muon_step_and_gradient_flow()`
:   Verify Muon updates 2D parameters and reduces loss.

`test_optimizer_registry_contains_muon()`
:   Verify Muon and MuonAdamW are registered in OPTIMIZER_REGISTRY.

`test_zeropower_via_newtonschulz5()`
:   Verify Newton-Schulz iteration produces approximately orthonormal matrices with singular values near 1.