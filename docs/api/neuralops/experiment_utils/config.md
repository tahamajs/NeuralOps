Module neuralops.experiment_utils.config
========================================

Functions
---------

`build_config_from_args() ‑> neuralops.experiment_utils.config.ExperimentConfig`
:   Build and return config from args.

`load_yaml_config(path: str) ‑> dict[str, typing.Any]`
:   Load yaml config from disk or config.

Classes
-------

`ExperimentConfig(model_name: str = 'distilbert-base-uncased', attn_type: str = 'full', method: str = 'baseline', compressor: str | None = None, task: str = 'sst2', max_len: int = 128, batch_size: int = 32, epochs: int = 5, lr: float = 1e-05, grad_accum: int = 1, seed: int = 42, output_dir: str = './experiments', log_dir: str = './logs', exp_name: str | None = None, trainer_type: str = 'hf', attn_implementation: str = 'full', lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1, rdrop_alpha: float = 5.0, r3f_epsilon: float = 1e-05, mixout_p: float = 0.9, llrd_decay: float = 0.95, dislora_beta: float = 0.5, dislora_capacity_factor: float = 1.0, log_base_dir: str = './logs', run_id: str | None = None, fisher_num_batches: int = 20, fisher_sample_size: int = 200, gelora_margin: float = 1.0, gelora_min_rank: int = 2, gelora_max_rank: int = 32, fisher_scale_max: float = 2.0, use_lion: bool = False, gradient_checkpointing: bool = False, early_stopping_patience: int | None = None, max_grad_norm: float = 1.0, num_seeds: int = 1, num_landmarks: int = 16, feature_dim: int = 128, k_early: int = 32, k_mid: int = 16, k_late: int = 12, use_gate: bool = True, progressive: bool = True, config: str | None = None, glue_base_path: str | None = None, compression_rank: int = 64)`
:   ExperimentConfig(model_name: str = 'distilbert-base-uncased', attn_type: str = 'full', method: str = 'baseline', compressor: str | None = None, task: str = 'sst2', max_len: int = 128, batch_size: int = 32, epochs: int = 5, lr: float = 1e-05, grad_accum: int = 1, seed: int = 42, output_dir: str = './experiments', log_dir: str = './logs', exp_name: str | None = None, trainer_type: str = 'hf', attn_implementation: str = 'full', lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1, rdrop_alpha: float = 5.0, r3f_epsilon: float = 1e-05, mixout_p: float = 0.9, llrd_decay: float = 0.95, dislora_beta: float = 0.5, dislora_capacity_factor: float = 1.0, log_base_dir: str = './logs', run_id: str | None = None, fisher_num_batches: int = 20, fisher_sample_size: int = 200, gelora_margin: float = 1.0, gelora_min_rank: int = 2, gelora_max_rank: int = 32, fisher_scale_max: float = 2.0, use_lion: bool = False, gradient_checkpointing: bool = False, early_stopping_patience: int | None = None, max_grad_norm: float = 1.0, num_seeds: int = 1, num_landmarks: int = 16, feature_dim: int = 128, k_early: int = 32, k_mid: int = 16, k_late: int = 12, use_gate: bool = True, progressive: bool = True, config: str | None = None, glue_base_path: str | None = None, compression_rank: int = 64)

    ### Instance variables

    `attn_implementation: str`
    :   The type of the None singleton.

    `attn_type: str`
    :   The type of the None singleton.

    `batch_size: int`
    :   The type of the None singleton.

    `compression_rank: int`
    :   The type of the None singleton.

    `compressor: str | None`
    :   The type of the None singleton.

    `config: str | None`
    :   The type of the None singleton.

    `dislora_beta: float`
    :   The type of the None singleton.

    `dislora_capacity_factor: float`
    :   The type of the None singleton.

    `early_stopping_patience: int | None`
    :   The type of the None singleton.

    `epochs: int`
    :   The type of the None singleton.

    `exp_name: str | None`
    :   The type of the None singleton.

    `feature_dim: int`
    :   The type of the None singleton.

    `fisher_num_batches: int`
    :   The type of the None singleton.

    `fisher_sample_size: int`
    :   The type of the None singleton.

    `fisher_scale_max: float`
    :   The type of the None singleton.

    `gelora_margin: float`
    :   The type of the None singleton.

    `gelora_max_rank: int`
    :   The type of the None singleton.

    `gelora_min_rank: int`
    :   The type of the None singleton.

    `glue_base_path: str | None`
    :   The type of the None singleton.

    `grad_accum: int`
    :   The type of the None singleton.

    `gradient_checkpointing: bool`
    :   The type of the None singleton.

    `k_early: int`
    :   The type of the None singleton.

    `k_late: int`
    :   The type of the None singleton.

    `k_mid: int`
    :   The type of the None singleton.

    `llrd_decay: float`
    :   The type of the None singleton.

    `log_base_dir: str`
    :   The type of the None singleton.

    `log_dir: str`
    :   The type of the None singleton.

    `lora_alpha: int`
    :   The type of the None singleton.

    `lora_dropout: float`
    :   The type of the None singleton.

    `lora_r: int`
    :   The type of the None singleton.

    `lr: float`
    :   The type of the None singleton.

    `max_grad_norm: float`
    :   The type of the None singleton.

    `max_len: int`
    :   The type of the None singleton.

    `method: str`
    :   The type of the None singleton.

    `mixout_p: float`
    :   The type of the None singleton.

    `model_name: str`
    :   The type of the None singleton.

    `num_landmarks: int`
    :   The type of the None singleton.

    `num_seeds: int`
    :   The type of the None singleton.

    `output_dir: str`
    :   The type of the None singleton.

    `progressive: bool`
    :   The type of the None singleton.

    `r3f_epsilon: float`
    :   The type of the None singleton.

    `rdrop_alpha: float`
    :   The type of the None singleton.

    `run_id: str | None`
    :   The type of the None singleton.

    `seed: int`
    :   The type of the None singleton.

    `task: str`
    :   The type of the None singleton.

    `trainer_type: str`
    :   The type of the None singleton.

    `use_gate: bool`
    :   The type of the None singleton.

    `use_lion: bool`
    :   The type of the None singleton.

    ### Methods

    `to_dict(self) ‑> dict[str, typing.Any]`
    :   To dict.