Module neuralops.config.config
==============================
Complete configuration dataclasses for all framework components.

Usage:
    cfg = Config()
    cfg.model.name = "distilbert-base-uncased"
    cfg.compression.rank = 64

Classes
-------

`CompressionConfig(attn_type: str = 'full', dropout: float = 0.1, rank: int = 32, svd_rank: int = 32, svd_energy: float = 0.95, min_rank: int = 4, max_rank: int = 64, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1, rdrop_alpha: float = 5.0, r3f_epsilon: float = 1e-05, llrd_decay: float = 0.95, mixout_p: float = 0.9, dislora_beta: float = 0.5, dislora_capacity_factor: float = 1.0, fisher_num_batches: int = 20, fisher_sample_size: int = 200, fisher_scale_max: float = 2.0, gelora_margin: float = 1.0, gelora_min_rank: int = 2, gelora_max_rank: int = 32, block_size: int = 64, local_window: int = 64, top_k_inter: int = 4, num_landmarks: int = 16, feature_dim: int = 128, rff_features: int = 128, k_early: int = 32, k_mid: int = 16, k_late: int = 12, use_gate: bool = True, progressive: bool = True, max_seq_len: int = 512, low_rank_cache_size: int = 16, cur_rank: int = 16, lsh_rounds: int = 4, lsh_bucket_size: int = 32, window_size: int = 128, compression_rank: int = 64, use_adaptive_gate: bool = True, gate_bias: float = 2.0, softpick_temperature: float = 1.0, rectify_interval: int = 4, aks_enabled: bool = True, rel_pos_max_distance: int = 128, use_rel_pos_bias: bool = True, adaptive_span: bool = True, max_span: float = 512.0, num_kernel_heads: int = 8, num_full_heads: int = 4, soa_rank: int = 32, svd_rank_qkv: int = 32, svd_rank_out: int = 32, num_local_heads: int = 8, num_global_heads: int = 4, num_experts: int = 8, top_k: int = 2, ratio: int = 9, n_heads: int | None = None, n_layers: int | None = None, use_full_attention: bool = False, use_fp8: bool = False, use_lion: bool = False, gradient_checkpointing: bool = False, early_stopping_patience: int | None = None, max_grad_norm: float = 1.0)`
:   All parameters related to attention injection, low‑rank compression,
    sparsity, hybrid heads, etc.
    These fields are read by the injection functions via getattr(comp_config, …).

    ### Instance variables

    `adaptive_span: bool`
    :   The type of the None singleton.

    `aks_enabled: bool`
    :   The type of the None singleton.

    `attn_type: str`
    :   The type of the None singleton.

    `block_size: int`
    :   The type of the None singleton.

    `compression_rank: int`
    :   The type of the None singleton.

    `cur_rank: int`
    :   The type of the None singleton.

    `dislora_beta: float`
    :   The type of the None singleton.

    `dislora_capacity_factor: float`
    :   The type of the None singleton.

    `dropout: float`
    :   The type of the None singleton.

    `early_stopping_patience: int | None`
    :   The type of the None singleton.

    `feature_dim: int`
    :   The type of the None singleton.

    `fisher_num_batches: int`
    :   The type of the None singleton.

    `fisher_sample_size: int`
    :   The type of the None singleton.

    `fisher_scale_max: float`
    :   The type of the None singleton.

    `gate_bias: float`
    :   The type of the None singleton.

    `gelora_margin: float`
    :   The type of the None singleton.

    `gelora_max_rank: int`
    :   The type of the None singleton.

    `gelora_min_rank: int`
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

    `local_window: int`
    :   The type of the None singleton.

    `lora_alpha: int`
    :   The type of the None singleton.

    `lora_dropout: float`
    :   The type of the None singleton.

    `lora_r: int`
    :   The type of the None singleton.

    `low_rank_cache_size: int`
    :   The type of the None singleton.

    `lsh_bucket_size: int`
    :   The type of the None singleton.

    `lsh_rounds: int`
    :   The type of the None singleton.

    `max_grad_norm: float`
    :   The type of the None singleton.

    `max_rank: int`
    :   The type of the None singleton.

    `max_seq_len: int`
    :   The type of the None singleton.

    `max_span: float`
    :   The type of the None singleton.

    `min_rank: int`
    :   The type of the None singleton.

    `mixout_p: float`
    :   The type of the None singleton.

    `n_heads: int | None`
    :   The type of the None singleton.

    `n_layers: int | None`
    :   The type of the None singleton.

    `num_experts: int`
    :   The type of the None singleton.

    `num_full_heads: int`
    :   The type of the None singleton.

    `num_global_heads: int`
    :   The type of the None singleton.

    `num_kernel_heads: int`
    :   The type of the None singleton.

    `num_landmarks: int`
    :   The type of the None singleton.

    `num_local_heads: int`
    :   The type of the None singleton.

    `progressive: bool`
    :   The type of the None singleton.

    `r3f_epsilon: float`
    :   The type of the None singleton.

    `rank: int`
    :   The type of the None singleton.

    `ratio: int`
    :   The type of the None singleton.

    `rdrop_alpha: float`
    :   The type of the None singleton.

    `rectify_interval: int`
    :   The type of the None singleton.

    `rel_pos_max_distance: int`
    :   The type of the None singleton.

    `rff_features: int`
    :   The type of the None singleton.

    `soa_rank: int`
    :   The type of the None singleton.

    `softpick_temperature: float`
    :   The type of the None singleton.

    `svd_energy: float`
    :   The type of the None singleton.

    `svd_rank: int`
    :   The type of the None singleton.

    `svd_rank_out: int`
    :   The type of the None singleton.

    `svd_rank_qkv: int`
    :   The type of the None singleton.

    `top_k: int`
    :   The type of the None singleton.

    `top_k_inter: int`
    :   The type of the None singleton.

    `use_adaptive_gate: bool`
    :   The type of the None singleton.

    `use_fp8: bool`
    :   The type of the None singleton.

    `use_full_attention: bool`
    :   The type of the None singleton.

    `use_gate: bool`
    :   The type of the None singleton.

    `use_lion: bool`
    :   The type of the None singleton.

    `use_rel_pos_bias: bool`
    :   The type of the None singleton.

    `window_size: int`
    :   The type of the None singleton.

`Config(model: neuralops.config.config.ModelConfig = <factory>, data: neuralops.config.config.DataConfig = <factory>, training: neuralops.config.config.TrainingConfig = <factory>, compression: neuralops.config.config.CompressionConfig = <factory>, method: neuralops.config.config.MethodConfig = <factory>, hardware: neuralops.config.config.HardwareConfig = <factory>, sweep: neuralops.config.config.SweepConfig = <factory>, experiment: neuralops.config.config.ExperimentConfig = <factory>)`
:   Master configuration that holds all sub‑configs.

    ### Instance variables

    `compression: neuralops.config.config.CompressionConfig`
    :   The type of the None singleton.

    `data: neuralops.config.config.DataConfig`
    :   The type of the None singleton.

    `experiment: neuralops.config.config.ExperimentConfig`
    :   The type of the None singleton.

    `hardware: neuralops.config.config.HardwareConfig`
    :   The type of the None singleton.

    `method: neuralops.config.config.MethodConfig`
    :   The type of the None singleton.

    `model: neuralops.config.config.ModelConfig`
    :   The type of the None singleton.

    `sweep: neuralops.config.config.SweepConfig`
    :   The type of the None singleton.

    `training: neuralops.config.config.TrainingConfig`
    :   The type of the None singleton.

`DataConfig(local_glue_path: str = '/Users/tahamajs/Documents/uni/datasets/glue', max_seq_length: int = 128, max_train_samples: int | None = None, max_val_samples: int | None = None, num_proc: int = 4, streaming: bool = False, remove_columns: list[str] = <factory>, split_mapping: dict[str, str] = <factory>)`
:   Data loading and preprocessing configuration.

    ### Instance variables

    `local_glue_path: str`
    :   The type of the None singleton.

    `max_seq_length: int`
    :   The type of the None singleton.

    `max_train_samples: int | None`
    :   The type of the None singleton.

    `max_val_samples: int | None`
    :   The type of the None singleton.

    `num_proc: int`
    :   The type of the None singleton.

    `remove_columns: list[str]`
    :   The type of the None singleton.

    `split_mapping: dict[str, str]`
    :   The type of the None singleton.

    `streaming: bool`
    :   The type of the None singleton.

`DatasetConfig(name: str, path: str, task_type: str = 'single', num_choices: int = 0, num_labels: int = 2)`
:   Description of a single dataset for an experiment sweep.

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

`DisLoRAConfig(beta: float = 0.5, capacity_factor: float = 1.0)`
:   DisLoRAConfig(beta: float = 0.5, capacity_factor: float = 1.0)

    ### Instance variables

    `beta: float`
    :   The type of the None singleton.

    `capacity_factor: float`
    :   The type of the None singleton.

`ExperimentConfig(base_models: dict[str, str] = <factory>, svd_ranks: list[int] = <factory>, datasets: list[neuralops.config.config.DatasetConfig] = <factory>, min_samples: int = 10, max_samples: int = 200, batch_size: int = 4, epochs: int = 1, learning_rate: float = 2e-05, max_length: int = 128, val_split: float = 0.2, seed: int = 42, gradient_accumulation_steps: int = 1, save_checkpoint_interval: int = 10, output_dir: str = './results', plots_dir: str = './plots/comprehensive', log_dir: str = './logs', selected_models: list[str] = <factory>, selected_datasets: list[str] = <factory>, selected_ranks: list[int] = <factory>)`
:   Multi‑run experiment grid (SVD ranks, datasets, models).

    ### Instance variables

    `base_models: dict[str, str]`
    :   The type of the None singleton.

    `batch_size: int`
    :   The type of the None singleton.

    `datasets: list[neuralops.config.config.DatasetConfig]`
    :   The type of the None singleton.

    `epochs: int`
    :   The type of the None singleton.

    `gradient_accumulation_steps: int`
    :   The type of the None singleton.

    `learning_rate: float`
    :   The type of the None singleton.

    `log_dir: str`
    :   The type of the None singleton.

    `max_length: int`
    :   The type of the None singleton.

    `max_samples: int`
    :   The type of the None singleton.

    `min_samples: int`
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

`FisherConfig(num_batches: int = 20, sample_size: int = 200, scale_max: float = 2.0)`
:   FisherConfig(num_batches: int = 20, sample_size: int = 200, scale_max: float = 2.0)

    ### Instance variables

    `num_batches: int`
    :   The type of the None singleton.

    `sample_size: int`
    :   The type of the None singleton.

    `scale_max: float`
    :   The type of the None singleton.

`GeLoRAConfig(margin: float = 1.0, min_rank: int = 2, max_rank: int = 32)`
:   GeLoRAConfig(margin: float = 1.0, min_rank: int = 2, max_rank: int = 32)

    ### Instance variables

    `margin: float`
    :   The type of the None singleton.

    `max_rank: int`
    :   The type of the None singleton.

    `min_rank: int`
    :   The type of the None singleton.

`HardwareConfig(device: str = 'auto', force_cpu_for_svd: bool = True, enable_fp16: bool = False, enable_bf16: bool = False, num_workers: int = 4, pin_memory: bool = False, torch_compile: bool = False, empty_cache_every_step: bool = False)`
:   HardwareConfig(device: str = 'auto', force_cpu_for_svd: bool = True, enable_fp16: bool = False, enable_bf16: bool = False, num_workers: int = 4, pin_memory: bool = False, torch_compile: bool = False, empty_cache_every_step: bool = False)

    ### Instance variables

    `device: str`
    :   The type of the None singleton.

    `empty_cache_every_step: bool`
    :   The type of the None singleton.

    `enable_bf16: bool`
    :   The type of the None singleton.

    `enable_fp16: bool`
    :   The type of the None singleton.

    `force_cpu_for_svd: bool`
    :   The type of the None singleton.

    `num_workers: int`
    :   The type of the None singleton.

    `pin_memory: bool`
    :   The type of the None singleton.

    `torch_compile: bool`
    :   The type of the None singleton.

`LLRDConfig(decay: float = 0.95)`
:   LLRDConfig(decay: float = 0.95)

    ### Instance variables

    `decay: float`
    :   The type of the None singleton.

`LoRAConfig(r: int = 8, alpha: int = 16, dropout: float = 0.1)`
:   LoRAConfig(r: int = 8, alpha: int = 16, dropout: float = 0.1)

    ### Instance variables

    `alpha: int`
    :   The type of the None singleton.

    `dropout: float`
    :   The type of the None singleton.

    `r: int`
    :   The type of the None singleton.

`MethodConfig(method: str = 'baseline', lora: neuralops.config.config.LoRAConfig = <factory>, rdrop: neuralops.config.config.RDropConfig = <factory>, r3f: neuralops.config.config.R3FConfig = <factory>, fisher: neuralops.config.config.FisherConfig = <factory>, gelora: neuralops.config.config.GeLoRAConfig = <factory>, llrd: neuralops.config.config.LLRDConfig = <factory>, mixout: neuralops.config.config.MixoutConfig = <factory>, dislora: neuralops.config.config.DisLoRAConfig = <factory>)`
:   Training method selection and its sub‑configs.

    ### Instance variables

    `dislora: neuralops.config.config.DisLoRAConfig`
    :   The type of the None singleton.

    `fisher: neuralops.config.config.FisherConfig`
    :   The type of the None singleton.

    `gelora: neuralops.config.config.GeLoRAConfig`
    :   The type of the None singleton.

    `llrd: neuralops.config.config.LLRDConfig`
    :   The type of the None singleton.

    `lora: neuralops.config.config.LoRAConfig`
    :   The type of the None singleton.

    `method: str`
    :   The type of the None singleton.

    `mixout: neuralops.config.config.MixoutConfig`
    :   The type of the None singleton.

    `r3f: neuralops.config.config.R3FConfig`
    :   The type of the None singleton.

    `rdrop: neuralops.config.config.RDropConfig`
    :   The type of the None singleton.

`MixoutConfig(p: float = 0.9)`
:   MixoutConfig(p: float = 0.9)

    ### Instance variables

    `p: float`
    :   The type of the None singleton.

`ModelConfig(name: str = 'distilbert-base-uncased', num_labels: int = 2, attention_type: str = 'full', use_fast_tokenizer: bool = True, trust_remote_code: bool = False, cache_dir: str = './cache', fix_layernorm_naming: bool = True, variants: list[str] = <factory>)`
:   Model loading configuration.

    ### Instance variables

    `attention_type: str`
    :   The type of the None singleton.

    `cache_dir: str`
    :   The type of the None singleton.

    `fix_layernorm_naming: bool`
    :   The type of the None singleton.

    `name: str`
    :   The type of the None singleton.

    `num_labels: int`
    :   The type of the None singleton.

    `trust_remote_code: bool`
    :   The type of the None singleton.

    `use_fast_tokenizer: bool`
    :   The type of the None singleton.

    `variants: list[str]`
    :   The type of the None singleton.

`R3FConfig(epsilon: float = 1e-05)`
:   R3FConfig(epsilon: float = 1e-05)

    ### Instance variables

    `epsilon: float`
    :   The type of the None singleton.

`RDropConfig(alpha: float = 5.0)`
:   RDropConfig(alpha: float = 5.0)

    ### Instance variables

    `alpha: float`
    :   The type of the None singleton.

`SweepConfig(name: str = 'sweep', algorithm: str = 'tpe', direction: str = 'maximize', metric: str = 'eval_accuracy', max_trials: int = 20, parallel_workers: int = 1, pruner: str = 'hyperband', search_space: dict[str, typing.Any] = <factory>, base_config: dict[str, typing.Any] = <factory>)`
:   Hyperparameter sweep configuration.

    ### Instance variables

    `algorithm: str`
    :   The type of the None singleton.

    `base_config: dict[str, typing.Any]`
    :   The type of the None singleton.

    `direction: str`
    :   The type of the None singleton.

    `max_trials: int`
    :   The type of the None singleton.

    `metric: str`
    :   The type of the None singleton.

    `name: str`
    :   The type of the None singleton.

    `parallel_workers: int`
    :   The type of the None singleton.

    `pruner: str`
    :   The type of the None singleton.

    `search_space: dict[str, typing.Any]`
    :   The type of the None singleton.

`TrainingConfig(batch_size: int = 16, eval_batch_size: int = 32, epochs: int = 3, learning_rate: float = 2e-05, weight_decay: float = 0.01, warmup_ratio: float = 0.1, warmup_steps: int = 0, gradient_accumulation_steps: int = 1, max_grad_norm: float = 1.0, lr_scheduler_type: str = 'linear', optimizer: str = 'adamw', early_stopping_patience: int | None = None, early_stopping_threshold: float = 0.001, save_strategy: str = 'epoch', save_steps: int = 500, save_total_limit: int = 2, save_only_model: bool = True, eval_strategy: str = 'epoch', eval_steps: int = 500, metric_for_best_model: str = 'eval_loss', greater_is_better: bool = False, fp16: bool = False, bf16: bool = False, report_to: str = 'none')`
:   Training hyperparameters.

    ### Instance variables

    `batch_size: int`
    :   The type of the None singleton.

    `bf16: bool`
    :   The type of the None singleton.

    `early_stopping_patience: int | None`
    :   The type of the None singleton.

    `early_stopping_threshold: float`
    :   The type of the None singleton.

    `epochs: int`
    :   The type of the None singleton.

    `eval_batch_size: int`
    :   The type of the None singleton.

    `eval_steps: int`
    :   The type of the None singleton.

    `eval_strategy: str`
    :   The type of the None singleton.

    `fp16: bool`
    :   The type of the None singleton.

    `gradient_accumulation_steps: int`
    :   The type of the None singleton.

    `greater_is_better: bool`
    :   The type of the None singleton.

    `learning_rate: float`
    :   The type of the None singleton.

    `lr_scheduler_type: str`
    :   The type of the None singleton.

    `max_grad_norm: float`
    :   The type of the None singleton.

    `metric_for_best_model: str`
    :   The type of the None singleton.

    `optimizer: str`
    :   The type of the None singleton.

    `report_to: str`
    :   The type of the None singleton.

    `save_only_model: bool`
    :   The type of the None singleton.

    `save_steps: int`
    :   The type of the None singleton.

    `save_strategy: str`
    :   The type of the None singleton.

    `save_total_limit: int`
    :   The type of the None singleton.

    `warmup_ratio: float`
    :   The type of the None singleton.

    `warmup_steps: int`
    :   The type of the None singleton.

    `weight_decay: float`
    :   The type of the None singleton.