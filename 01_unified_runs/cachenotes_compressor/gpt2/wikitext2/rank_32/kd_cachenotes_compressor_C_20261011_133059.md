# Benchmark Run Summary: cachenotes_compressor

- **Run ID**: `kd_cachenotes_compressor_C_20261011_133059`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T10:03:52.013140+00:00
- **Git Commit**: `e0119bde06` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `cachenotes_compressor` |
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
| **Peak VRAM** | 2522.14 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module cachenotes_compressor --model gpt2 --dataset wikitext2 --rank 32
```
