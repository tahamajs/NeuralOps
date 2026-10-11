# Benchmark Run Summary: huffman_coder

- **Run ID**: `kd_huffman_coder_C_20261011_145548`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T11:28:32.066648+00:00
- **Git Commit**: `0f67bb39bf` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `huffman_coder` |
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
| **Peak VRAM** | 2516.76 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module huffman_coder --model gpt2 --dataset wikitext2 --rank 32
```
