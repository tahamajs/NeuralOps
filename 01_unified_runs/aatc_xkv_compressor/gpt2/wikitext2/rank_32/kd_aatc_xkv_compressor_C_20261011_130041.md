# Benchmark Run Summary: aatc_xkv_compressor

- **Run ID**: `kd_aatc_xkv_compressor_C_20261011_130041`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T09:33:29.455165+00:00
- **Git Commit**: `a2276b9f36` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `aatc_xkv_compressor` |
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
| **Peak VRAM** | 2708.26 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module aatc_xkv_compressor --model gpt2 --dataset wikitext2 --rank 32
```
