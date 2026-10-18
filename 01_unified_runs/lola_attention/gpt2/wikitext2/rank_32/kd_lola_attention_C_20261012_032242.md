# Benchmark Run Summary: lola_attention

- **Run ID**: `kd_lola_attention_C_20261012_032242`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T03:27:12.800433+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `lola_attention` |
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
| **Peak VRAM** | 2784.54 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module lola_attention --model gpt2 --dataset wikitext2 --rank 32
```
