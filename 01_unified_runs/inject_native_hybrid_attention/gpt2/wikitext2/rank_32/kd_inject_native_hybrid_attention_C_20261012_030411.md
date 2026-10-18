# Benchmark Run Summary: inject_native_hybrid_attention

- **Run ID**: `kd_inject_native_hybrid_attention_C_20261012_030411`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T03:07:12.456329+00:00
- **Git Commit**: `619f535ab3` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `inject_native_hybrid_attention` |
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
| **Peak VRAM** | 2845.87 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module inject_native_hybrid_attention --model gpt2 --dataset wikitext2 --rank 32
```
