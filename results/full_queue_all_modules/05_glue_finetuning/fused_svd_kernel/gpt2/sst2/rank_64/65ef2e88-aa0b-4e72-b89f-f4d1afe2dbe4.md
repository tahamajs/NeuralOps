# Benchmark Run Summary: fused_svd_kernel

- **Run ID**: `65ef2e88-aa0b-4e72-b89f-f4d1afe2dbe4`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-14T05:24:46Z
- **Git Commit**: `32e519631c` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `fused_svd_kernel` |
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
| **Accuracy** | 0.5000 |
| **F1 Score** | 0.0000 |
| **Loss** | 0.6864 |
| **Duration (s)** | 12.22s |
| **Peak VRAM** | 803.01025390625 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module fused_svd_kernel --model gpt2 --dataset sst2 --rank 64
```
