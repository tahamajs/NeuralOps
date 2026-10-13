# Benchmark Run Summary: hybrid_attention

- **Run ID**: `kd_hybrid_attention_C_20261013_080551`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-13T09:29:20.181157+00:00
- **Git Commit**: `13a8e2e2ff` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `hybrid_attention` |
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
| **Peak VRAM** | 5963.01 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module hybrid_attention --model gpt2 --dataset wikitext2 --rank 32
```
