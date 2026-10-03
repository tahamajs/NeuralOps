Module neuralops.evaluation.eval_perplexity
===========================================
Evaluate Module Perplexity (WikiText-2) on GPT-2.

Production-hardened to produce results strictly comparable to run_gpt2_sweep.py:
  1. Builds real calibration DataLoader from WikiText-2 train split (500 samples, batch_size=4, max_length=128)
  2. Passes calibration_loader AND tokenizer to apply_module_to_model
  3. Replaces module_type="auto" with explicit ATTENTION_REGISTRY / COMPRESSOR_REGISTRY lookup
  4. Uses fp16 on CUDA (torch_dtype=torch.float16 if cuda else torch.float32)
  5. Uses full test split without arbitrary max_samples truncation (or explicitly records partial eval)
  6. Captures peak VRAM, elapsed time, params before/after, compression rate
  7. Records seq_len/stride from actual arguments
  8. Adds is_baseline flag when module_name == "none" or "full"
  9. Standard python logging replacing raw print()
  10. Saves per-window NLLs to JSON for downstream variance analysis
  11. Robust CPU fallback reducing seq_len if CUDA OOM occurs

Functions
---------

`build_calibration_loader(train_dataset, tokenizer, num_samples: int = 500, batch_size: int = 4, max_length: int = 128) ‑> list[dict[str, torch.Tensor]]`
:   Build a list of batched dictionary samples from WikiText-2 train split.

`compute_perplexity(model: nn.Module, tokenizer: Any, test_dataset: Any, seq_len: int = 1024, stride: int = 512, max_eval_tokens: int | None = None, device: str = 'cuda') ‑> tuple[float, dict[str, typing.Any]]`
:   Compute sliding-window perplexity identical to run_gpt2_sweep.py.

`evaluate_module_ppl(model_path: str, module_name: str, output_dir: str, dataset_name: str = 'wikitext-2-raw-v1', split: str = 'test', rank: int = 64, seq_len: int = 1024, stride: int = 512, max_eval_tokens: int | None = None) ‑> dict[str, typing.Any]`
:   Load base checkpoint, inject module with calibration data, compute PPL and telemetry.