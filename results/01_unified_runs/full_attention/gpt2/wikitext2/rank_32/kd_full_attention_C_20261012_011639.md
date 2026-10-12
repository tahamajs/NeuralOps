# Benchmark Run Summary: full_attention

- **Run ID**: `kd_full_attention_C_20261012_011639`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T01:20:09.634864+00:00
- **Git Commit**: `a3376cf3b4` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `full_attention` |
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
| **Peak VRAM** | 2840.97 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module full_attention --model gpt2 --dataset wikitext2 --rank 32
```
