# Benchmark Run Summary: padre_attention

- **Run ID**: `kd_padre_attention_C_20261012_042948`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T04:33:01.509954+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `padre_attention` |
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
| **Peak VRAM** | 2840.77 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module padre_attention --model gpt2 --dataset wikitext2 --rank 32
```
