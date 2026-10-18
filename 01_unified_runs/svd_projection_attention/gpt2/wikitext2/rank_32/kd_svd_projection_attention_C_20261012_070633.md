# Benchmark Run Summary: svd_projection_attention

- **Run ID**: `kd_svd_projection_attention_C_20261012_070633`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T07:09:13.771983+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `svd_projection_attention` |
| **Model** | `gpt2` |
| **Dataset** | `wikitext2` |
| **Rank** | `32` |
| **Epochs** | `1` |
| **Batch Size** | `8` |

## Primary Metrics
| Metric | Value |
|---|---|
| **Accuracy** | N/A |
| **F1 Score** | N/A |
| **Loss** | N/A |
| **Duration (s)** | 0.0s |
| **Peak VRAM** | 2781.04 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module svd_projection_attention --model gpt2 --dataset wikitext2 --rank 32
```
