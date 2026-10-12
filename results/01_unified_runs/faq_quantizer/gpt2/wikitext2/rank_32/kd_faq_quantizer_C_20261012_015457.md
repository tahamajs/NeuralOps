# Benchmark Run Summary: faq_quantizer

- **Run ID**: `kd_faq_quantizer_C_20261012_015457`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T01:55:46.309103+00:00
- **Git Commit**: `20116686f5` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `faq_quantizer` |
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
| **Peak VRAM** | 862.44 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module faq_quantizer --model gpt2 --dataset wikitext2 --rank 32
```
