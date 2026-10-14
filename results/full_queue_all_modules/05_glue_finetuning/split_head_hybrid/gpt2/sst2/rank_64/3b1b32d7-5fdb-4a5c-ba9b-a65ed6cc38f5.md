# Benchmark Run Summary: split_head_hybrid

- **Run ID**: `3b1b32d7-5fdb-4a5c-ba9b-a65ed6cc38f5`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-14T05:52:59Z
- **Git Commit**: `32e519631c` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `split_head_hybrid` |
| **Model** | `gpt2` |
| **Dataset** | `sst2` |
| **Rank** | `64` |
| **Training Type** | `finetune` |
| **Batch Size** | `2` |

## Primary Metrics
| Metric | Value |
|---|---|
| **Pre-KD Perplexity** | N/A |
| **Post-KD Perplexity** | N/A |
| **Recovery %** | N/A |
| **KD Gain** | N/A |
| **Accuracy** | 0.5000 |
| **F1 Score** | 0.0000 |
| **Loss** | 0.7178 |
| **Duration (s)** | 13.25s |
| **Peak VRAM** | 841.26611328125 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module split_head_hybrid --model gpt2 --dataset sst2 --rank 64
```
