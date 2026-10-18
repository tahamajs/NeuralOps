# Benchmark Run Summary: compressed_mha_attention

- **Run ID**: `kd_compressed_mha_attention_C_20261011_222221`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T22:25:53.782212+00:00
- **Git Commit**: `671b506b86` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `compressed_mha_attention` |
| **Model** | `gpt2` |
| **Dataset** | `wikitext2` |
| **Rank** | `32` |
| **Epochs** | `1` |
| **Batch Size** | `12` |

## Primary Metrics
| Metric | Value |
|---|---|
| **Accuracy** | N/A |
| **F1 Score** | N/A |
| **Loss** | N/A |
| **Duration (s)** | 0.0s |
| **Peak VRAM** | 4525.85 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module compressed_mha_attention --model gpt2 --dataset wikitext2 --rank 32
```
