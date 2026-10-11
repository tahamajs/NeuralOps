# Benchmark Run Summary: astro_quantizer

- **Run ID**: `kd_astro_quantizer_C_20261011_132417`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T09:56:58.917349+00:00
- **Git Commit**: `363d9d95c7` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `astro_quantizer` |
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
| **Peak VRAM** | 2519.06 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module astro_quantizer --model gpt2 --dataset wikitext2 --rank 32
```
