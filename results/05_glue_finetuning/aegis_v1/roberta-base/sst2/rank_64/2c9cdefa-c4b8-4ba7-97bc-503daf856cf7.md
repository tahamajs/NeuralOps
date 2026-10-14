# Benchmark Run Summary: aegis_v1

- **Run ID**: `2c9cdefa-c4b8-4ba7-97bc-503daf856cf7`
- **Status**: **FAILED**
- **Timestamp**: 2026-10-13T19:55:14Z
- **Git Commit**: `7443010e6a` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `aegis_v1` |
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
| **Accuracy** | N/A |
| **F1 Score** | N/A |
| **Loss** | N/A |
| **Duration (s)** | 0.0s |
| **Peak VRAM** | 0.0 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module aegis_v1 --model roberta-base --dataset sst2 --rank 64
```
