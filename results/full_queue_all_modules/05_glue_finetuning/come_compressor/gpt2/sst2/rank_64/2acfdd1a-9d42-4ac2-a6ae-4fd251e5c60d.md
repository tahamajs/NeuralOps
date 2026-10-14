# Benchmark Run Summary: come_compressor

- **Run ID**: `2acfdd1a-9d42-4ac2-a6ae-4fd251e5c60d`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-14T05:16:29Z
- **Git Commit**: `32e519631c` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `come_compressor` |
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
| **Loss** | 4.0217 |
| **Duration (s)** | 11.24s |
| **Peak VRAM** | 627.48291015625 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module come_compressor --model gpt2 --dataset sst2 --rank 64
```
