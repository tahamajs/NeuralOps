# Benchmark Run Summary: lola_attention

- **Run ID**: `eb019bdc-fcf0-442f-af18-e2ab42a9d73b`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-13T23:27:54Z
- **Git Commit**: `14e65f4ff6` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `lola_attention` |
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
| **Accuracy** | 0.5300 |
| **F1 Score** | 0.6846 |
| **Loss** | 0.6783 |
| **Duration (s)** | 73.86s |
| **Peak VRAM** | 911.720703125 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module lola_attention --model roberta-base --dataset sst2 --rank 64
```
