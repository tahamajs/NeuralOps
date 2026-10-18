# Benchmark Run Summary: kronecker_tt_lora

- **Run ID**: `kd_kronecker_tt_lora_C_20261012_095007`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T09:53:49.044949+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `kronecker_tt_lora` |
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
| **Peak VRAM** | 2837.56 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module kronecker_tt_lora --model gpt2 --dataset wikitext2 --rank 32
```
