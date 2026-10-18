# Benchmark Run Summary: lora_attention

- **Run ID**: `kd_lora_attention_C_20261012_032724`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T03:30:15.825643+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `lora_attention` |
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
| **Peak VRAM** | 2843.27 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module lora_attention --model gpt2 --dataset wikitext2 --rank 32
```
