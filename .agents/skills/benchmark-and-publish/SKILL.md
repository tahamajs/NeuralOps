---
name: benchmark-and-publish
description: Runs a module's run_all_4_<module>.py GPU benchmark, applies the quality gate (status PASS and accuracy >= threshold), and — only on pass — packages and uploads the complete model plus full metrics and a publication-ready model card to the private Hugging Face repo, updates the module's documentation, and commits/pushes results to a dedicated benchmark-results-<module> git branch. Use this whenever the user asks to "run the benchmark", "run it on the GPU server", "upload to Hugging Face", "publish results", "commit the results", or gives a queue of modules to benchmark one by one. Also trigger when they ask to change shared run settings (batch size, recovery steps, which models to include) that should apply to all subsequent runs.
---

# Benchmark, Quality-Gate, and Publish

Run exactly **one module at a time**. Never launch a second GPU benchmark while one is still
running — it causes memory pressure and corrupts timing/VRAM metrics for both.

## Step 0 — Confirm current run settings before launching

Before the first run in a session (or whenever the user changes a setting), confirm the
active configuration and make sure it applies uniformly to every subsequent module, not just
the next one:
- Model(s) to include (default convention in this repo: `roberta` only, once the user has
  scoped it down — verify with them rather than assuming all 4 models).
- `--recovery_steps` (commonly 50 or 200 — confirm which).
- `--batch_size` (confirm the current value; if the user says "use batch size N going
  forward," update the shared default in `Codes/framework/run_quality_sweep.py` — don't just
  pass `--batch_size N` on this one invocation and let the default silently revert for the
  next module).
- Precision (fp16), learning rate (commonly 2e-4), LoRA rank/alpha.

If a setting changes mid-queue, apply it going forward only — don't retroactively invalidate
already-published results unless the user explicitly asks you to re-run them.

## Step 1 — Launch and monitor

```
ssh gpu "powershell -Command \"cd D:\\svd_project; python -u run_all_4_<module>.py --models <models> --recovery_steps <n> --batch_size <n>\""
```

Poll periodically rather than blocking silently. If a run stalls (no CPU-time progress across
two consecutive checks, or a recurrent/sequential-scan module is clearly going to time out),
kill it, purge its partial checkpoint directory, and report the failure — don't let a single
stuck module block the whole queue indefinitely.

## Step 2 — Apply the quality gate

- **PASS** = reported status is `PASS` **and** accuracy ≥ the agreed threshold (commonly 56%
  for SST-2 binary classification — confirm the number in use if unsure).
- **PASS_LOW / FAIL / TIMEOUT / NaN loss** → reject. Delete the checkpoint directory
  immediately. Do **not** upload to Hugging Face and do **not** create a git branch for it.
  Logs/metrics may be kept for debugging if useful, but clearly marked as a failed run.

## Step 3 — On PASS, package and upload the *complete* artifact set

Don't upload only LoRA adapter weights — upload the full model. For each passing run, produce:

1. **Full model files**: `config.json`, full weights (`pytorch_model.bin`/`model.safetensors`
   or an explicit `torch.save(model.state_dict(), ...)` of the complete model — base + any
   injected attention/compression module + LoRA), tokenizer files
   (`tokenizer_config.json`, `vocab.json`/`merges.txt`, `special_tokens_map.json`).
2. **`metrics.json` + `summary.csv`**: accuracy, F1 (binary/macro), precision, recall,
   validation loss; total/trainable parameter counts and compression ratio; FLOPs reduction if
   available; number of injected/compressed/fallback layers; average and per-layer
   reconstruction error/rank if applicable; peak VRAM, tokens/sec, elapsed time; GPU name, CUDA
   version, PyTorch version; full hyperparameters; timestamp and git commit hash.
3. **Attention maps / activations** (if the module supports `output_attentions` or hooks are
   feasible): mean attention per layer at minimum, saved under
   `results/{module}/{model}/attention_maps/` and `.../activations/` — a small sample subset
   (e.g. 4–10 validation examples) is fine; document the subset size used.
4. **`run.log`**, **`environment.txt`** (`pip freeze`/`nvidia-smi` output).
5. **Comprehensive model card (`README.md`)**: YAML front matter (tags, dataset, metrics) +
   architecture/module description + intended use + full hyperparameters + the metrics table
   from (2) + a reproducibility code snippet to load and verify the weights + links to the
   GitHub commit/branch and the original base model.

Upload to the **private** repo `Hooshaai/svd-linear-attention-{model}-{module}` via
`huggingface_hub`/`push_to_hub(private=True)`. If the repo already exists, update files rather
than erroring. Verify the upload by listing the repo's files afterward — don't assume success
from a lack of exception.

## Step 4 — Update documentation

Update (or create, via the `doc-code-alignment` skill's section structure if it doesn't yet
exist) `Docs/content/attentions/<module>.md` or `compressions/<module>.md` with: the benchmark
results table for this model/run, the Hugging Face repo link, paths to any saved attention
maps/activations, and the full hyperparameters used.

## Step 5 — Commit and push, one branch per module

```bash
git checkout -b benchmark-results-<module>   # or checkout if it exists
git add results/<module-related-summary-files> Docs/content/.../<module>.md
git commit -m "benchmark(<module>): record <model> benchmark results (<accuracy>% acc, PASS)"
git push origin benchmark-results-<module>
git checkout main   # always return to main — never leave a results branch checked out
```

Never commit large model binaries to git — those live on Hugging Face only. Only commit
summary CSVs/JSON, documentation, and small config artifacts.

## Step 6 — Report and move to the next module

One-line status per module: name, model, steps, accuracy, PASS/REJECTED, HF repo URL (if
uploaded), git branch (if committed). Then proceed to the next module in the queue —
sequentially, never in parallel, to protect GPU memory and timing accuracy.

## End-of-queue summary

After the full queue finishes, produce one final table: module | baseline accuracy | achieved
accuracy | Δ accuracy | F1 | peak VRAM | HF link (successful only). Flag any module that was
rejected/timed out so it can be revisited with adjusted settings later.
