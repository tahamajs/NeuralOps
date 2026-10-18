# Benchmark Run Summary: come_compressor

- **Run ID**: `kd_come_compressor_C_20261011_133417`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T10:08:03.764918+00:00
- **Git Commit**: `e0119bde06` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `come_compressor` |
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
| **Peak VRAM** | 2182.04 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module come_compressor --model gpt2 --dataset wikitext2 --rank 32
```
