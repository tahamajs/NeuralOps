# Benchmark Run Summary: mamba_svd_flash

- **Run ID**: `kd_mamba_svd_flash_C_20261012_163552`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T16:54:03.820758+00:00
- **Git Commit**: `acc158890b` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `mamba_svd_flash` |
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
| **Peak VRAM** | 1829.25 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module mamba_svd_flash --model gpt2 --dataset wikitext2 --rank 32
```
