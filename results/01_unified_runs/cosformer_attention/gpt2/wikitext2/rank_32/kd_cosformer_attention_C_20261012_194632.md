# Benchmark Run Summary: cosformer_attention

- **Run ID**: `kd_cosformer_attention_C_20261012_194632`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T19:47:17.462922+00:00
- **Git Commit**: `4904ad5364` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `cosformer_attention` |
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
| **Peak VRAM** | 2882.82 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module cosformer_attention --model gpt2 --dataset wikitext2 --rank 32
```
