# Benchmark Run Summary: rotational_slicegpt_int4

- **Run ID**: `kd_rotational_slicegpt_int4_C_20261011_161023`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T12:43:17.108283+00:00
- **Git Commit**: `0f67bb39bf` (dirty=True)

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
| **Peak VRAM** | 2517.31 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module rotational_slicegpt_int4 --model gpt2 --dataset wikitext2 --rank 32
```
