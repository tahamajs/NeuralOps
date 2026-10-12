# Benchmark Run Summary: linformer_attention

- **Run ID**: `kd_linformer_attention_C_20261012_152149`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T15:26:13.803098+00:00
- **Git Commit**: `6d9b481f76` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `linformer_attention` |
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
| **Peak VRAM** | 2818.92 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module linformer_attention --model gpt2 --dataset wikitext2 --rank 32
```
