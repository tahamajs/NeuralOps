# Benchmark Run Summary: soa_attn

- **Run ID**: `0ddfaa45-02c1-4acd-95cb-db3ba45456da`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-14T07:05:28Z
- **Git Commit**: `e8d52e8170` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `soa_attn` |
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
| **Accuracy** | 0.6767 |
| **F1 Score** | 0.5941 |
| **Loss** | 0.5858 |
| **Duration (s)** | 24.63s |
| **Peak VRAM** | 893.541015625 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module soa_attn --model roberta-base --dataset sst2 --rank 64
```
