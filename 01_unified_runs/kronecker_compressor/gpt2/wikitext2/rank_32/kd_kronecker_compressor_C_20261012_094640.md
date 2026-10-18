# Benchmark Run Summary: kronecker_compressor

- **Run ID**: `kd_kronecker_compressor_C_20261012_094640`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T09:49:52.287952+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `kronecker_compressor` |
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
| **Peak VRAM** | 2713.74 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module kronecker_compressor --model gpt2 --dataset wikitext2 --rank 32
```
