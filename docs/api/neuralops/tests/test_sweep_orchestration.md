Module neuralops.tests.test_sweep_orchestration
===============================================
Unit tests for sweep orchestration, checkpoint resolution, plateau detection, and resumption.

Classes
-------

`TestSweepOrchestrationLogic()`
:   Verify sweep runner decision branches, plateau handling, and summary.csv operations.

    ### Methods

    `test_batch_size_escalation_oom_handler(self)`
    :   Verify OOM simulation halves batch size until min_batch_size.

    `test_causal_module_skip_logic_for_encoder(self)`
    :   Verify early skip correctly tags causal module on roberta before running GPU steps.

    `test_parent_checkpoint_explicit_cli_override(self)`
    :   Verify --parent_checkpoint CLI argument explicitly takes precedence over defaults.

    `test_plateau_detection_logic(self)`
    :   Verify that baseline 50.92% accuracy after recovery triggers plateau classification.

    `test_summary_csv_resumption_skips_existing(self)`
    :   Verify sweep reads existing completed modules from summary.csv and skips them.