# Benchmark Run Summary: token_compressed_attention

- **Run ID**: `kd_token_compressed_attention_C_20261012_071500`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T07:19:20.136436+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `token_compressed_attention` |
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
| **Peak VRAM** | 2841.42 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module token_compressed_attention --model gpt2 --dataset wikitext2 --rank 32
```
