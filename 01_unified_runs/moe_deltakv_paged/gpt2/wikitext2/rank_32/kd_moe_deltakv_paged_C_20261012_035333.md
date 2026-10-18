# Benchmark Run Summary: moe_deltakv_paged

- **Run ID**: `kd_moe_deltakv_paged_C_20261012_035333`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T03:58:34.992003+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `moe_deltakv_paged` |
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
| **Peak VRAM** | 2841.09 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module moe_deltakv_paged --model gpt2 --dataset wikitext2 --rank 32
```
