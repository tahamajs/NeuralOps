# Benchmark Run Summary: aegis_spectron_fp8

- **Run ID**: `kd_aegis_spectron_fp8_C_20261012_175030`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T17:52:26.124959+00:00
- **Git Commit**: `4b230f0d85` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `aegis_spectron_fp8` |
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
| **Peak VRAM** | 1665.83 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module aegis_spectron_fp8 --model gpt2 --dataset wikitext2 --rank 32
```
