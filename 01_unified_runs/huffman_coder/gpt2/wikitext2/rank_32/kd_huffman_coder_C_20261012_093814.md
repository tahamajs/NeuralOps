# Benchmark Run Summary: huffman_coder

- **Run ID**: `kd_huffman_coder_C_20261012_093814`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T09:40:31.222400+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `huffman_coder` |
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
| **Peak VRAM** | 3094.01 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module huffman_coder --model gpt2 --dataset wikitext2 --rank 32
```
