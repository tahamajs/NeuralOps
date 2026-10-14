# Benchmark Run Summary: mixattention_moska

- **Run ID**: `2bba5b22-6fba-45eb-8fcf-2efeb17358e0`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-13T23:35:31Z
- **Git Commit**: `14e65f4ff6` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `mixattention_moska` |
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
| **Accuracy** | 0.7667 |
| **F1 Score** | 0.8108 |
| **Loss** | 0.4934 |
| **Duration (s)** | 68.48s |
| **Peak VRAM** | 995.2138671875 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module mixattention_moska --model roberta-base --dataset sst2 --rank 64
```
