# Benchmark Run Summary: aegis_v5

- **Run ID**: `dcfc10d1-49c9-49dd-8ea8-efb61a8a0cec`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-14T05:13:37Z
- **Git Commit**: `32e519631c` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `aegis_v5` |
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
| **Duration (s)** | 21.14s |
| **Peak VRAM** | 836.51611328125 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module aegis_v5 --model gpt2 --dataset sst2 --rank 64
```
