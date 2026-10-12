# Benchmark Run Summary: faq_quantizer

- **Run ID**: `kd_faq_quantizer_C_20261012_091301`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T09:16:24.966514+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `faq_quantizer` |
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
| **Peak VRAM** | 2850.52 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module faq_quantizer --model gpt2 --dataset wikitext2 --rank 32
```
