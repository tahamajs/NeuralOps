Module neuralops.benchmarks.run_seqlen_scaling_bench
====================================================
Sequence Length Scaling Benchmark Harness.

Measures:
  1. Throughput (tokens/sec)
  2. Latency (ms/token)
  3. Peak VRAM (MB)
across sequence lengths [128, 256, 512, 1024] at batch_size=1 on GPT-2.
Runs each measurement 3 times and reports the median.
Handles OOM gracefully.
Outputs:
  - results/seqlen_scaling/<MODULE>/summary.csv
  - results/06_adhoc_experiments/SEQLEN_SCALING_MASTER.csv

Functions
---------

`main()`
:   

`measure_single_length(model: torch.nn.modules.module.Module, seq_len: int, batch_size: int, device: torch.device, num_trials: int = 3, num_iters: int = 25, warmup_iters: int = 5) ‑> dict[str, typing.Any]`
:   Measures throughput, latency, and peak VRAM for a single sequence length.
    Runs `num_trials` times and returns the median results.

`run_benchmark_for_module(base_model: torch.nn.modules.module.Module, base_snapshot: neuralops.core.snapshot_restore.ModelStateSnapshot, module_name: str, group_name: str, seq_lens: list[int], batch_size: int, device: torch.device, output_dir: str, master_csv: str, num_trials: int = 3, num_iters: int = 25, warmup_iters: int = 5) ‑> list[dict[str, typing.Any]]`
:   Evaluates one module across all specified sequence lengths.
    Restores baseline state before adapting.

`save_results(module_name: str, group_name: str, results: list[dict[str, typing.Any]], output_dir: str, master_csv: str)`
:   Saves individual module summary.csv and appends to master CSV.