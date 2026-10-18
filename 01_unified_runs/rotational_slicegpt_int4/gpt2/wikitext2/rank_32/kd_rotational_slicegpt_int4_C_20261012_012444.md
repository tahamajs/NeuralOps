# Benchmark Run Summary: rotational_slicegpt_int4

- **Run ID**: `kd_rotational_slicegpt_int4_C_20261012_012444`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T01:24:55.248862+00:00
- **Git Commit**: `47f3447b65` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `rotational_slicegpt_int4` |
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
| **Peak VRAM** | 581.99 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module rotational_slicegpt_int4 --model gpt2 --dataset wikitext2 --rank 32
```
