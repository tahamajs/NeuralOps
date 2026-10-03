Module neuralops.tests.test_lora
================================
Comprehensive unit tests for native LoRA module.

Classes
-------

`TestApplyLoRAOnModels()`
:   Test suite for apply_lora on multi-layer transformer structures.

    ### Methods

    `test_apply_lora_to_distilbert_style(self)`
    :

    `test_apply_lora_to_synthetic_qwen_block(self)`
    :

    `test_save_and_load_checkpoint(self)`
    :

`TestLoRACompressorIntegration()`
:   Verify LoRA adapters integrate properly with compressor recovery fine-tuning.

    ### Methods

    `test_compressor_unmerged_lora_preservation(self)`
    :   Verify that LoRA adapters remain unmerged and trainable alongside a compressed base_layer.

    `test_recovery_parameter_selection_precision(self)`
    :   Verify targeted recovery parameter selection freezes backbone & norms while enabling LoRA & head.

    `test_recovery_training_loss_reduction(self)`
    :   Verify a 5-step recovery loop decreases cross-entropy loss on a synthetic task.

`TestLoRALinear()`
:   Test suite for the standalone LoRALinear layer.

    ### Methods

    `test_gradient_flow_and_backprop(self)`
    :   Gradients must flow into lora_A and lora_B, but not base weights.

    `test_initialization_and_forward_shape(self)`
    :

    `test_merge_and_unmerge(self)`
    :   Verify weights merge into base_layer and unmerge correctly.

    `test_parameter_freezing(self)`
    :   Base weights must be frozen, only lora_A and lora_B trainable.

    `test_zero_initialization_identity(self)`
    :   At step 0, LoRA output must be exactly identical to base linear output.