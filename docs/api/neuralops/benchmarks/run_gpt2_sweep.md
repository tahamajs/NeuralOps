Module neuralops.benchmarks.run_gpt2_sweep
==========================================
Decoder Benchmark Runner: Comprehensive evaluation of attention/compression modules
on GPT-2 with WikiText-2.

Measures per module:
  - Perplexity (PPL) via sliding-window evaluation
  - Peak VRAM, latency, throughput
  - Parameter savings / compression ratio
  - Injection / compression statistics (real work done, not just PASS)

Production features :
  1.  Upsert-based CSV: one row per module, no duplicates across runs.
  2.  Intelligent skip: only skips modules that ACTUALLY did work
      (num_injected_attention_layers > 0 or num_compressed_layers > 0).
  3.  --no_skip / --force / --only_failed / --retry_flags flags.
  4.  Baseline PPL caching (re-use across runs unless --recompute_baseline).
  5.  Full metric capture: injection counts, recovery time, LoRA params, etc.
  6.  Phantom-pass detection: PASS_WITHOUT_EFFECT status.
  7.  tqdm progress bar with per-module timing.
  8.  Structured logging to file + console (with optional colored output).
  9.  Atomic CSV writes via tmp-file + rename.
  10. Multi-worker safety: single global summary + per-worker CSV.
  11. Comprehensive diag.json with dtype, shape, device info on failure.
  12. Token packing (no padding waste) for calibration.
  13. Configurable calibration size.
  14. Reproducible seeding per module.
  15. Graceful KeyboardInterrupt: saves partial results before exiting.
  16. Per-module JSON result with full stats.
  17. End-of-run Markdown report generation.
  18. Baseline sanity check: warns if baseline PPL is anomalous.
  19. Memory-leak guard: explicit gc + cuda.empty_cache between modules.
  20. Config file support (YAML/JSON).
  21. --dry_run preview with expected I/O.
  22. Better exception messages with stack locals on failure (optional).
  23. Multi-GPU aware (--device cuda:N).
  24. Summary statistics at the end (mean, median, best/worst).
  25. Auto-detection of deprecated module files from multiple locations.

Usage:
  python decoder_benchmark.py --modules all --no_skip --force
  python decoder_benchmark.py --modules mla,svd_llm --only_failed
  python decoder_benchmark.py --total_workers 4 --worker_id 0

Functions
---------

`build_calibration_loader(train_dataset, tokenizer, num_samples: int = 500, batch_size: int = 4, max_length: int = 128) ‑> list[dict[str, torch.Tensor]]`
:   Pack short WikiText lines into fixed-length chunks (no padding waste).
    Returns a list of batches: [{"input_ids": (B, T), "attention_mask": (B, T)}, ...]

`classify_status(ppl_ratio: float) ‑> str`
:   

`compute_perplexity_with_metrics(model: nn.Module, tokenizer, test_dataset, seq_len: int = 1024, stride: int = 512, max_eval_tokens: int | None = None, device: str = 'cuda') ‑> dict[str, typing.Any]`
:   Sliding-window perplexity (GPT-2 / LLaMA-style).
    
    PPL = exp( Σ NLL_t × target_len_t / Σ target_len_t )

`get_all_decoder_modules(category: str = 'all') ‑> list[str]`
:   Dynamically discover all GPT-2-compatible attention and compressor modules.

`is_real_success(status: str, row: dict[str, Any]) ‑> bool`
:   A module is a 'real success' iff it PASSED AND did measurable work.

`load_deprecated_modules() ‑> set[str]`
:   Collect deprecated module names from all known file locations.

`main(argv: list[str] | None = None)`
:   

`parse_args(argv: list[str] | None = None) ‑> argparse.Namespace`
:   

`print_summary_stats(rows: dict[str, dict[str, str]])`
:   

`run_kd_recovery(student: nn.Module, teacher: nn.Module, tokenizer, train_dataset, device: str = 'cuda', steps: int = 100, seq_len: int = 128, lr: float = 2.5e-05, kl_temp: float = 4.0) ‑> float`
:   Lightweight KD recovery (Method A / logits KD) on train_dataset.
    Returns recovery time in seconds.

`setup_logging(log_file: str | None, verbose: bool = False)`
:   

`upsert_row(csv_path: Path, row: dict[str, Any], lock_path: Path | None = None)`
:   Atomic upsert by module_name. Uses FileLock for multiprocess safety.

`write_diag(diag_path: Path, module: str, exc: BaseException, extra: dict[str, Any] | None = None, include_locals: bool = False)`
:   

`write_markdown_report(out_dir: Path, csv_path: Path, baseline_ppl: float)`
:   

Classes
-------

`ResultRow(module_name: str, status: str = 'PENDING', perplexity: float = -1.0, baseline_ppl: float = -1.0, delta_ppl: float = 0.0, ppl_ratio: float = 1.0, compression_rate: float = 1.0, compression_ratio_pct: float = 0.0, total_params_before: int = 0, total_params_after: int = 0, delta_params: int = 0, num_injected_attention_layers: int = 0, num_compressed_layers: int = 0, num_fallback_layers: int = 0, lora_params: int = 0, calibration_time_s: float = 0.0, recovery_time_s: float = 0.0, peak_vram_mb: float = 0.0, latency_ms_per_token: float = 0.0, throughput_tok_per_sec: float = 0.0, eval_elapsed_s: float = 0.0, wall_elapsed_s: float = 0.0, error: str = '', timestamp: str = '', post_kd_ppl: float = -1.0, delta_ppl_recovery: float = 0.0)`
:   ResultRow(module_name: 'str', status: 'str' = 'PENDING', perplexity: 'float' = -1.0, baseline_ppl: 'float' = -1.0, delta_ppl: 'float' = 0.0, ppl_ratio: 'float' = 1.0, compression_rate: 'float' = 1.0, compression_ratio_pct: 'float' = 0.0, total_params_before: 'int' = 0, total_params_after: 'int' = 0, delta_params: 'int' = 0, num_injected_attention_layers: 'int' = 0, num_compressed_layers: 'int' = 0, num_fallback_layers: 'int' = 0, lora_params: 'int' = 0, calibration_time_s: 'float' = 0.0, recovery_time_s: 'float' = 0.0, peak_vram_mb: 'float' = 0.0, latency_ms_per_token: 'float' = 0.0, throughput_tok_per_sec: 'float' = 0.0, eval_elapsed_s: 'float' = 0.0, wall_elapsed_s: 'float' = 0.0, error: 'str' = '', timestamp: 'str' = '', post_kd_ppl: 'float' = -1.0, delta_ppl_recovery: 'float' = 0.0)

    ### Instance variables

    `baseline_ppl: float`
    :   The type of the None singleton.

    `calibration_time_s: float`
    :   The type of the None singleton.

    `compression_rate: float`
    :   The type of the None singleton.

    `compression_ratio_pct: float`
    :   The type of the None singleton.

    `delta_params: int`
    :   The type of the None singleton.

    `delta_ppl: float`
    :   The type of the None singleton.

    `delta_ppl_recovery: float`
    :   The type of the None singleton.

    `error: str`
    :   The type of the None singleton.

    `eval_elapsed_s: float`
    :   The type of the None singleton.

    `latency_ms_per_token: float`
    :   The type of the None singleton.

    `lora_params: int`
    :   The type of the None singleton.

    `module_name: str`
    :   The type of the None singleton.

    `num_compressed_layers: int`
    :   The type of the None singleton.

    `num_fallback_layers: int`
    :   The type of the None singleton.

    `num_injected_attention_layers: int`
    :   The type of the None singleton.

    `peak_vram_mb: float`
    :   The type of the None singleton.

    `perplexity: float`
    :   The type of the None singleton.

    `post_kd_ppl: float`
    :   The type of the None singleton.

    `ppl_ratio: float`
    :   The type of the None singleton.

    `recovery_time_s: float`
    :   The type of the None singleton.

    `status: str`
    :   The type of the None singleton.

    `throughput_tok_per_sec: float`
    :   The type of the None singleton.

    `timestamp: str`
    :   The type of the None singleton.

    `total_params_after: int`
    :   The type of the None singleton.

    `total_params_before: int`
    :   The type of the None singleton.

    `wall_elapsed_s: float`
    :   The type of the None singleton.

    ### Methods

    `to_csv_dict(self) ‑> dict[str, typing.Any]`
    :