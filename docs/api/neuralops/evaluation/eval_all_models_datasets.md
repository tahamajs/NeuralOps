Module neuralops.evaluation.eval_all_models_datasets
====================================================

Functions
---------

`apply_module_to_model(model: nn.Module, module_name: str, module_type: str = 'auto', rank: int = 16, device: torch.device = device(type='cpu'), calibration_data: Any | None = None, calibration_loader: Any | None = None, tokenizer: Any | None = None) ‑> tuple[torch.nn.modules.module.Module, float]`
:   Inject or compress the model.
    
    FIX-1: forwards calibration_loader + tokenizer to compressors.
    FIX-2: removed unreachable candidate=None.
    FIX-11: no random fake calibration.

`build_calibration_loader(dataset: str, model_family: str, seq_len: int, device: torch.device, n_samples: int = 500, batch_size: int = 8)`
:   FIX-1: build a real calibration DataLoader from the training split.
    Returns (loader, tokenizer, source_str) or (None, None, None).

`build_model_instance(model_family: str, num_labels: int = 2, hidden_size: int = 128, num_heads: int = 4, num_layers: int = 2, vocab_size: int | None = None, seq_len: int = 32, device: torch.device = device(type='cpu')) ‑> tuple[torch.nn.modules.module.Module, typing.Any]`
:   

`evaluate_single_combination(track: str, module_name: str, module_type: str, model_family: str, dataset: str, rank: int = 16, num_warmup: int = 2, num_steps: int = 5, batch_size: int = 128, seq_len: int = 256, device: torch.device = device(type='cpu'), results_dir: str = './results/gpu_server', allow_synthetic: bool = False, skip_deprecated: bool = True) ‑> neuralops.evaluation.eval_all_models_datasets.BenchmarkResult`
:   

`evaluate_single_combination_opt(track: str, module_name: str, module_type: str, model_family: str, dataset: str, rank: int = 16, num_warmup: int = 2, num_steps: int = 5, batch_size: int = 128, seq_len: int = 256, device: torch.device = device(type='cpu'), results_dir: str = './results/gpu_server', allow_synthetic: bool = False, skip_deprecated: bool = True) ‑> neuralops.evaluation.eval_all_models_datasets.BenchmarkResult`
:   

`get_base_model_cached(model_family: str, num_labels: int = 2, hidden_size: int = 128, num_heads: int = 4, num_layers: int = 2, device: torch.device = device(type='cpu')) ‑> tuple[torch.nn.modules.module.Module, typing.Any]`
:   FIX-5: replace deepcopy with fresh build + state_dict load.
    Avoids 2x memory and deepcopy failures.

`get_dataset_batch(dataset_name: str, model_family: str, batch_size: int = 128, seq_len: int = 256, vocab_size: int = 30522, num_labels: int = 2, device: torch.device = device(type='cpu'), allow_synthetic: bool = False) ‑> dict[str, torch.Tensor]`
:   

`probe_optimal_batch_size(model: nn.Module, model_family: str, dataset: str, seq_len: int, device: torch.device, target_vram_ratio: float = 0.35, max_batch_cap: int = 64, min_batch_floor: int = 4, allow_synthetic: bool = False) ‑> int`
:   

`run_track(track: str, model_family: str, dataset: str, rank: int = 64, seq_len: int = 128, batch_size: int = 8, device: torch.device | None = None, results_dir: str = './results', allow_synthetic: bool = False) ‑> list[neuralops.evaluation.eval_all_models_datasets.BenchmarkResult]`
:   Run every module in the given track against the given model/dataset.

Classes
-------

`BenchmarkResult(track: str, module_name: str, module_type: str, model_family: str, dataset: str, status: str, error_message: str | None = None, forward_latency_ms: float = 0.0, backward_latency_ms: float = 0.0, total_step_latency_ms: float = 0.0, throughput_samples_per_sec: float = 0.0, speedup_vs_baseline: float = 1.0, param_count: int = 0, trainable_param_count: int = 0, compression_ratio: float = 1.0, peak_memory_mb: float = 0.0, memory_savings_pct: float = 0.0, loss: float = 0.0, accuracy_pct: float = 0.0, f1_score: float = 0.0, perplexity: float = 0.0, grad_norm: float = 0.0, mcc: float = 0.0, pearson: float = 0.0, spearman: float = 0.0, fidelity_score: float = 0.0, flops_macs: float = 0.0, throughput_tokens_sec: float = 0.0, latency_variance_ms: float = 0.0, ci_lower: float | None = None, ci_upper: float | None = None, p_value_vs_baseline: float | None = None, mean_seed_acc: float | None = None, std_seed_acc: float | None = None, calibration_used: bool = False, calibration_source: str | None = None, num_compressed_layers: int = 0, num_fallback_layers: int = 0, avg_reconstruction_error: float | None = None, params_before: int = 0, params_after: int = 0, trajectories: dict[str, list[float]] = <factory>, timestamp: str = <factory>)`
:   BenchmarkResult(track: 'str', module_name: 'str', module_type: 'str', model_family: 'str', dataset: 'str', status: 'str', error_message: 'str | None' = None, forward_latency_ms: 'float' = 0.0, backward_latency_ms: 'float' = 0.0, total_step_latency_ms: 'float' = 0.0, throughput_samples_per_sec: 'float' = 0.0, speedup_vs_baseline: 'float' = 1.0, param_count: 'int' = 0, trainable_param_count: 'int' = 0, compression_ratio: 'float' = 1.0, peak_memory_mb: 'float' = 0.0, memory_savings_pct: 'float' = 0.0, loss: 'float' = 0.0, accuracy_pct: 'float' = 0.0, f1_score: 'float' = 0.0, perplexity: 'float' = 0.0, grad_norm: 'float' = 0.0, mcc: 'float' = 0.0, pearson: 'float' = 0.0, spearman: 'float' = 0.0, fidelity_score: 'float' = 0.0, flops_macs: 'float' = 0.0, throughput_tokens_sec: 'float' = 0.0, latency_variance_ms: 'float' = 0.0, ci_lower: 'float | None' = None, ci_upper: 'float | None' = None, p_value_vs_baseline: 'float | None' = None, mean_seed_acc: 'float | None' = None, std_seed_acc: 'float | None' = None, calibration_used: 'bool' = False, calibration_source: 'str | None' = None, num_compressed_layers: 'int' = 0, num_fallback_layers: 'int' = 0, avg_reconstruction_error: 'float | None' = None, params_before: 'int' = 0, params_after: 'int' = 0, trajectories: 'dict[str, list[float]]' = <factory>, timestamp: 'str' = <factory>)

    ### Instance variables

    `accuracy_pct: float`
    :   The type of the None singleton.

    `avg_reconstruction_error: float | None`
    :   The type of the None singleton.

    `backward_latency_ms: float`
    :   The type of the None singleton.

    `calibration_source: str | None`
    :   The type of the None singleton.

    `calibration_used: bool`
    :   The type of the None singleton.

    `ci_lower: float | None`
    :   The type of the None singleton.

    `ci_upper: float | None`
    :   The type of the None singleton.

    `compression_ratio: float`
    :   The type of the None singleton.

    `dataset: str`
    :   The type of the None singleton.

    `error_message: str | None`
    :   The type of the None singleton.

    `f1_score: float`
    :   The type of the None singleton.

    `fidelity_score: float`
    :   The type of the None singleton.

    `flops_macs: float`
    :   The type of the None singleton.

    `forward_latency_ms: float`
    :   The type of the None singleton.

    `grad_norm: float`
    :   The type of the None singleton.

    `latency_variance_ms: float`
    :   The type of the None singleton.

    `loss: float`
    :   The type of the None singleton.

    `mcc: float`
    :   The type of the None singleton.

    `mean_seed_acc: float | None`
    :   The type of the None singleton.

    `memory_savings_pct: float`
    :   The type of the None singleton.

    `model_family: str`
    :   The type of the None singleton.

    `module_name: str`
    :   The type of the None singleton.

    `module_type: str`
    :   The type of the None singleton.

    `num_compressed_layers: int`
    :   The type of the None singleton.

    `num_fallback_layers: int`
    :   The type of the None singleton.

    `p_value_vs_baseline: float | None`
    :   The type of the None singleton.

    `param_count: int`
    :   The type of the None singleton.

    `params_after: int`
    :   The type of the None singleton.

    `params_before: int`
    :   The type of the None singleton.

    `peak_memory_mb: float`
    :   The type of the None singleton.

    `pearson: float`
    :   The type of the None singleton.

    `perplexity: float`
    :   The type of the None singleton.

    `spearman: float`
    :   The type of the None singleton.

    `speedup_vs_baseline: float`
    :   The type of the None singleton.

    `status: str`
    :   The type of the None singleton.

    `std_seed_acc: float | None`
    :   The type of the None singleton.

    `throughput_samples_per_sec: float`
    :   The type of the None singleton.

    `throughput_tokens_sec: float`
    :   The type of the None singleton.

    `timestamp: str`
    :   The type of the None singleton.

    `total_step_latency_ms: float`
    :   The type of the None singleton.

    `track: str`
    :   The type of the None singleton.

    `trainable_param_count: int`
    :   The type of the None singleton.

    `trajectories: dict[str, list[float]]`
    :   The type of the None singleton.

    ### Methods

    `to_dict(self) ‑> dict[str, typing.Any]`
    :