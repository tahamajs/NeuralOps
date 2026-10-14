# Benchmark Run Summary: mamba_svd_flash

- **Run ID**: `7fef19ef-aee3-49ed-8694-49ff0fa3bf3f`
- **Status**: **FAILED**
- **Timestamp**: 2026-10-13T23:32:31Z
- **Git Commit**: `14e65f4ff6` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `mamba_svd_flash` |
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
python -m neuralops.benchmarks run --module mamba_svd_flash --model roberta-base --dataset sst2 --rank 64
```
