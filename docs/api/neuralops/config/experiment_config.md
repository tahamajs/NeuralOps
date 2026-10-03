Module neuralops.config.experiment_config
=========================================

Classes
-------

`DatasetConfig(name: str, path: str, task_type: str, num_choices: int = 0, num_labels: int = 2)`
:   DatasetConfig(name: str, path: str, task_type: str, num_choices: int = 0, num_labels: int = 2)

    ### Instance variables

    `name: str`
    :   The type of the None singleton.

    `num_choices: int`
    :   The type of the None singleton.

    `num_labels: int`
    :   The type of the None singleton.

    `path: str`
    :   The type of the None singleton.

    `task_type: str`
    :   The type of the None singleton.

`ExperimentConfig(base_models: dict[str, str] = <factory>, svd_ranks: list[int] = <factory>, datasets: list[neuralops.config.experiment_config.DatasetConfig] = <factory>, min_samples: int = 10, max_samples: int = 200, batch_size: int = 4, epochs: int = 1, learning_rate: float = 2e-05, max_length: int = 128, val_split: float = 0.2, seed: int = 42, gradient_accumulation_steps: int = 1, optimizer_type: str = 'adamw', lr_scheduler_type: str = 'linear', warmup_steps: int = 0, weight_decay: float = 0.01, mixed_precision: str = 'no', gradient_checkpointing: bool = False, max_grad_norm: float = 1.0, save_checkpoint_interval: int = 10, output_dir: str = './results', plots_dir: str = './plots/comprehensive', log_dir: str = './logs', selected_models: list[str] = <factory>, selected_datasets: list[str] = <factory>, selected_ranks: list[int] = <factory>)`
:   ExperimentConfig(base_models: dict[str, str] = <factory>, svd_ranks: list[int] = <factory>, datasets: list[neuralops.config.experiment_config.DatasetConfig] = <factory>, min_samples: int = 10, max_samples: int = 200, batch_size: int = 4, epochs: int = 1, learning_rate: float = 2e-05, max_length: int = 128, val_split: float = 0.2, seed: int = 42, gradient_accumulation_steps: int = 1, optimizer_type: str = 'adamw', lr_scheduler_type: str = 'linear', warmup_steps: int = 0, weight_decay: float = 0.01, mixed_precision: str = 'no', gradient_checkpointing: bool = False, max_grad_norm: float = 1.0, save_checkpoint_interval: int = 10, output_dir: str = './results', plots_dir: str = './plots/comprehensive', log_dir: str = './logs', selected_models: list[str] = <factory>, selected_datasets: list[str] = <factory>, selected_ranks: list[int] = <factory>)

    ### Instance variables

    `base_models: dict[str, str]`
    :   The type of the None singleton.

    `batch_size: int`
    :   The type of the None singleton.

    `datasets: list[neuralops.config.experiment_config.DatasetConfig]`
    :   The type of the None singleton.

    `epochs: int`
    :   The type of the None singleton.

    `gradient_accumulation_steps: int`
    :   The type of the None singleton.

    `gradient_checkpointing: bool`
    :   The type of the None singleton.

    `learning_rate: float`
    :   The type of the None singleton.

    `log_dir: str`
    :   The type of the None singleton.

    `lr_scheduler_type: str`
    :   The type of the None singleton.

    `max_grad_norm: float`
    :   The type of the None singleton.

    `max_length: int`
    :   The type of the None singleton.

    `max_samples: int`
    :   The type of the None singleton.

    `min_samples: int`
    :   The type of the None singleton.

    `mixed_precision: str`
    :   The type of the None singleton.

    `optimizer_type: str`
    :   The type of the None singleton.

    `output_dir: str`
    :   The type of the None singleton.

    `plots_dir: str`
    :   The type of the None singleton.

    `save_checkpoint_interval: int`
    :   The type of the None singleton.

    `seed: int`
    :   The type of the None singleton.

    `selected_datasets: list[str]`
    :   The type of the None singleton.

    `selected_models: list[str]`
    :   The type of the None singleton.

    `selected_ranks: list[int]`
    :   The type of the None singleton.

    `svd_ranks: list[int]`
    :   The type of the None singleton.

    `val_split: float`
    :   The type of the None singleton.

    `warmup_steps: int`
    :   The type of the None singleton.

    `weight_decay: float`
    :   The type of the None singleton.