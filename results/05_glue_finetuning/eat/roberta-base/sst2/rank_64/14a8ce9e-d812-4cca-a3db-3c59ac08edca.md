# Benchmark Run Summary: eat

- **Run ID**: `14a8ce9e-d812-4cca-a3db-3c59ac08edca`
- **Status**: **FAILED**
- **Timestamp**: 2026-10-13T21:46:21Z
- **Git Commit**: `7443010e6a` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `eat` |
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
python -m neuralops.benchmarks run --module eat --model roberta-base --dataset sst2 --rank 64
```
