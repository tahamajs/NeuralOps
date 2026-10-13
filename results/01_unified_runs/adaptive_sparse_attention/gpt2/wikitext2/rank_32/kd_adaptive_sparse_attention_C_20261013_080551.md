# Benchmark Run Summary: adaptive_sparse_attention

- **Run ID**: `kd_adaptive_sparse_attention_C_20261013_080551`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-13T08:35:19.570925+00:00
- **Git Commit**: `728c797cda` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `adaptive_sparse_attention` |
| **Model** | `gpt2` |
| **Dataset** | `wikitext2` |
| **Rank** | `32` |
| **Epochs** | `1` |
| **Batch Size** | `4` |

## Primary Metrics
| Metric | Value |
|---|---|
| **Accuracy** | N/A |
| **F1 Score** | N/A |
| **Loss** | N/A |
| **Duration (s)** | 0.0s |
| **Peak VRAM** | 3211.05 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module adaptive_sparse_attention --model gpt2 --dataset wikitext2 --rank 32
```
