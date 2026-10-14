# Benchmark Run Summary: hamming_attention

- **Run ID**: `1ed4def7-72a5-461a-bdc6-ed58bd4d5cae`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-13T22:10:44Z
- **Git Commit**: `7443010e6a` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `hamming_attention` |
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
| **Accuracy** | 0.7233 |
| **F1 Score** | 0.7025 |
| **Loss** | 0.5433 |
| **Duration (s)** | 71.18s |
| **Peak VRAM** | 1013.23876953125 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module hamming_attention --model roberta-base --dataset sst2 --rank 64
```
