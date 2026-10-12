# Benchmark Run Summary: rpca_compressor

- **Run ID**: `kd_rpca_compressor_C_20261012_104128`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T10:44:30.738772+00:00
- **Git Commit**: `d4f0b5f899` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `rpca_compressor` |
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
| **Peak VRAM** | 2681.69 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module rpca_compressor --model gpt2 --dataset wikitext2 --rank 32
```
