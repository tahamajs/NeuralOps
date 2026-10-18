# Benchmark Run Summary: kronecker_tt_lora

- **Run ID**: `kd_kronecker_tt_lora_C_20261012_012336`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T01:24:30.324280+00:00
- **Git Commit**: `47f3447b65` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `kronecker_tt_lora` |
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
| **Peak VRAM** | 920.87 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module kronecker_tt_lora --model gpt2 --dataset wikitext2 --rank 32
```
