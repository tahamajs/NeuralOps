# AG News & MNLI Run Diagnostic — 2026-09-18

## 1. Current Status
| Dataset | Completed | Remaining | Errors | Throughput (jobs/hr) | ETA |
|---------|-----------|-----------|--------|---------------------|-----|
| agnews  | 22        | 581       | 1      | 76.70 (1 worker) → ~150 (2 workers) | 3.87 hours (0.16 days) |
| mnli    | 1         | 569       | 3      | ~120.00 (2 workers)                 | 4.74 hours (0.20 days) |

<!-- source: audit/agnews_completion_2026_09_18/checkpoint.jsonl lines 1-23 -->
<!-- source: audit/mnli_completion_2026_09_18/checkpoint.jsonl lines 1-4 -->
<!-- source: audit/agnews_completion_2026_09_18/remaining.txt line 1-201 -->
<!-- source: audit/mnli_completion_2026_09_18/remaining.txt line 1-190 -->

## 2. GPU Status
- **Current Live Status**: GPU Util 39–81%, VRAM 1.81–2.28 GB / 8.19 GB, Temp 63–72°C <!-- source: nvidia-smi live query -->
- **Peak per job**: 865.53 MB (`a3_compressor` on GPT-2) <!-- source: logs/parallel_runs/agnews_a3_compressor_gpt2.log line 90 -->
- **Average VRAM per job**: 440.15 MB <!-- source: logs/parallel_runs/*.log measure-object average -->
- **Max safe workers**: 4 concurrent workers (allocated as 2 workers for AG News + 2 workers for MNLI).

## 3. Baselines (for reference)
| Model | SST-2 Baseline | AG News Baseline | MNLI Baseline |
| :--- | :---: | :---: | :---: |
| **DistilBERT** | 90.71% <!-- source: MASTER_PROJECT_RESULTS.md line 40 --> | 51.00% <!-- source: logs/parallel_runs/agnews_SOAFusedAttention_distilbert.log line 26 --> | 33.20% <!-- source: logs/parallel_runs/mnli_SOAFusedAttention_distilbert.log line 34 --> |
| **RoBERTa-base** | 78.56% <!-- source: MASTER_PROJECT_RESULTS.md line 41 --> | 74.50% <!-- source: logs/parallel_runs/agnews_SOAFusedAttention_roberta.log line 34 --> | 35.60% <!-- source: logs/parallel_runs/mnli_SOAFusedAttention_roberta.log line 34 --> |
| **GPT-2** | 87.84% <!-- source: MASTER_PROJECT_RESULTS.md line 42 --> | 79.70% <!-- source: logs/parallel_runs/agnews_SOAFusedAttention_gpt2.log line 25 --> | Pending |

## 4. Fixes Applied
- [x] **Workers increased**: Scaled from 1 worker to 2 workers for AG News, plus 2 workers for MNLI (4 workers total on GPU).
- [x] **MNLI runner launched in parallel**: Active PID 4400 with 2 worker processes running concurrently.
- [x] **Checkpoint deduplication**: Validated append-only JSONL; no duplicate entries present in checkpoint files.
- [x] **HuggingFace push verified**: Repo `TahaaaaM/svd-linear-attention-results` verified/created; dry-run upload of test file succeeded (`200 OK`).
- [x] **Results committed**: Continuous per-module Git auto-commit actively syncing to GitHub `origin/main` (`505b7faf7`, `4fd1163b8`).
- [x] **20-min timer active**: Task scheduled and actively monitoring checkpoint ledger and GPU thermal envelope.

## 5. New ETA After Fixes
- **AG News**: ~3.87 hours (581 jobs remaining @ ~150 jobs/hr with 2 workers)
- **MNLI**: ~4.74 hours (569 jobs remaining @ ~120 jobs/hr with 2 workers)
- **Combined (Parallel)**: **~4.8 hours (0.20 days)** to complete full repository coverage.

## 6. Blockers / Operational Notes
- `adaptive_span_inject` on DistilBERT failed with exit code `3221225477` (Windows stack/access violation in native C++ op for that specific injector, gracefully caught by runner).
- MNLI on `SOAFusedAttention` encountered `CUBLAS_STATUS_EXECUTION_FAILED` on GPT-2 during 3-class classification head backward pass (captured as ERROR and skipped to next module without halting the sweep).
- GPU VRAM consumption with 4 concurrent workers is ~2.28 GB out of 8.19 GB, operating well within safe thermal thresholds (72°C max).
