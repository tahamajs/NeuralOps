# Benchmark Run Summary: svd_compressed

- **Run ID**: `52ab2eea-3c64-4fdd-bf10-e9ae4725e6fd`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-14T07:08:33Z
- **Git Commit**: `e1a6047e43` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `svd_compressed` |
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
| **Accuracy** | 0.6933 |
| **F1 Score** | 0.6567 |
| **Loss** | 0.5654 |
| **Duration (s)** | 37.11s |
| **Peak VRAM** | 912.294921875 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module svd_compressed --model roberta-base --dataset sst2 --rank 64
```
