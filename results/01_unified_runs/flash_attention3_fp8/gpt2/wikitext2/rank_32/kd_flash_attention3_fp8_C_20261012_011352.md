# Benchmark Run Summary: flash_attention3_fp8

- **Run ID**: `kd_flash_attention3_fp8_C_20261012_011352`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T01:16:15.501768+00:00
- **Git Commit**: `f08bf07296` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `flash_attention3_fp8` |
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
| **Peak VRAM** | 2837.27 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module flash_attention3_fp8 --model gpt2 --dataset wikitext2 --rank 32
```
