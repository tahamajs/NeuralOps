# Benchmark Run Summary: hamming_attention

- **Run ID**: `kd_hamming_attention_C_20261012_020245`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T02:10:47.856189+00:00
- **Git Commit**: `117be82d03` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `hamming_attention` |
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
| **Peak VRAM** | 2839.34 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module hamming_attention --model gpt2 --dataset wikitext2 --rank 32
```
