Module neuralops.benchmarks.run_quality_sweep
=============================================
Cached Qwen sequence-classification quality sweep.

Fine-tunes the Qwen classification head first, then evaluates each canonical
attention/compression module from the same fine-tuned checkpoint.

Debugged v2 — 15 fixes applied:
1. override_to now merged into effective_timeout
2. Skip logic enabled for PASS statuses
3. dataset_dir sentinel removed (only appended when set)
4. datetime.utcnow() → datetime.now(timezone.utc)
5. Unified FIELDS schema (no duplicate)
6. perf_metrics/comp_metrics initialized to None before try
7. NaN loss aborts the batch (continue)
8. p.data.half() wrapped in no_grad + dtype check
9. torch.set_num_threads respects OMP_NUM_THREADS
10. Duplicate lora import removed
11. Dead multiprocessing code removed
12. SLOW_MODULE_TIMEOUT_OVERRIDES unified at module level
13. existing set actually used for skip
14. Dead config code removed
15. Step loss accumulation clarified

Functions
---------

`clear_vram_between_modules()`
:   

`collect_and_export_full_metrics(model, eval_batches, model_name, seq_len=128, device=None, fp16=True, out_dir='results', perf_metrics=None, comp_metrics=None, status='PASS', elapsed_s=0.0)`
:   

`compute_bh_fdr(raw_p: list[float]) ‑> list[float]`
:   

`compute_bootstrap_ci(accs: list[float], n_bootstraps: int = 500, ci_level: float = 0.95) ‑> tuple[float, float]`
:   

`compute_compression_metrics(model: torch.nn.Module, module_name: str, module_type: str, total_params_before: int, total_params_after: int, trainable_recovery_params: int = 0) ‑> dict[str, typing.Any]`
:   

`compute_confidence_interval_95(accs: list[float]) ‑> tuple[float, float]`
:   

`compute_t_test_p_value(accs1: list[float], accs2: list[float]) ‑> float`
:   

`encode(tokenizer, batch, dataset_name, seq_len, device)`
:   

`evaluate(model, tokenizer, ds, name, seq_len, batch_size, device, fp16=True, precomputed_batches=None)`
:   

`evaluate_comprehensive(model, tokenizer, ds, name, seq_len, batch_size, device, fp16=True, precomputed_batches=None)`
:   

`evaluate_perplexity(model, tokenizer, ds, seq_len, batch_size, device, fp16=True)`
:   

`fields(name: str) ‑> tuple[str, str | None]`
:   

`finetune(model, tokenizer, train, name, steps, seq_len, batch_size, device, fp16, lr=0.0001)`
:   

`finetune_lora(model, tokenizer, train, name, steps, seq_len, batch_size, device, fp16, lr=0.0002, timeout=600, precomputed_batches=None, optimizer_type='adamw', muon_lr=0.02, adam_lr=None, weight_decay=0.01, span_lambda=0.0)`
:   Fine-tune LoRA adapters and sequence classification head.
    
    Fix 7: NaN loss aborts the batch instead of propagating.
    Fix 15: step loss accumulation is explicit.

`finetune_with_kd(model, teacher_model, tokenizer, train, name, steps, seq_len, batch_size, device, fp16, lr=5e-05, alpha=0.5, temperature=2.0, precomputed_batches=None)`
:   

`main()`
:   

`modules() ‑> list[str]`
:   

`parse_args(argv=None)`
:   

`prepare_eval_batches(tokenizer, ds, name, seq_len, batch_size, device)`
:   

`prepare_train_batches(tokenizer, ds, name, seq_len, batch_size, device, max_batches=10)`
:   

`run_module_isolated(module_name: str, args, timeout_s: int) ‑> dict`
:   

`run_single_module_main(args) ‑> dict`
: