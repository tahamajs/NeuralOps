# Benchmark Run Summary: granite_hybrid

- **Run ID**: `a740e6c4-ddad-4dee-88d9-a318f8fbe88a`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-14T13:14:25Z
- **Git Commit**: `unknown` (dirty=False)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `granite_hybrid` |
| **Model** | `roberta-base` |
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
| **Loss** | 0.7121 |
| **Duration (s)** | 16.83s |
| **Peak VRAM** | 3318.59375 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module granite_hybrid --model roberta-base --dataset sst2 --rank 64
```
