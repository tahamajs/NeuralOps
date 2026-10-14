# Benchmark Run Summary: aegis_hybrid

- **Run ID**: `d2ecbacd-50af-46bb-a133-7cbf82b8eb83`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-14T05:10:48Z
- **Git Commit**: `32e519631c` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `aegis_hybrid` |
| **Model** | `gpt2` |
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
| **Accuracy** | 0.6500 |
| **F1 Score** | 0.6957 |
| **Loss** | 0.6598 |
| **Duration (s)** | 18.93s |
| **Peak VRAM** | 799.25634765625 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module aegis_hybrid --model gpt2 --dataset sst2 --rank 64
```
