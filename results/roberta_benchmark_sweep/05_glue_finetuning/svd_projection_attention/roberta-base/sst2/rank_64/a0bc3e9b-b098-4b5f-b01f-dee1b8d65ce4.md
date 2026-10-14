# Benchmark Run Summary: svd_projection_attention

- **Run ID**: `a0bc3e9b-b098-4b5f-b01f-dee1b8d65ce4`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-14T07:12:41Z
- **Git Commit**: `e1a6047e43` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `svd_projection_attention` |
| **Model** | `roberta-base` |
| **Dataset** | `sst2` |
| **Rank** | `64` |
| **Training Type** | `finetune` |
| **Batch Size** | `16` |

## Primary Metrics
| Metric | Value |
|---|---|
| **Pre-KD Perplexity** | N/A |
| **Post-KD Perplexity** | N/A |
| **Recovery %** | N/A |
| **KD Gain** | N/A |
| **Accuracy** | 0.6967 |
| **F1 Score** | 0.7479 |
| **Loss** | 0.6179 |
| **Duration (s)** | 24.87s |
| **Peak VRAM** | 902.294921875 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module svd_projection_attention --model roberta-base --dataset sst2 --rank 64
```
