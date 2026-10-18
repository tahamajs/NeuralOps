# Benchmark Run Summary: hla_attention

- **Run ID**: `kd_hla_attention_C_20261012_180007`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T18:01:53.741017+00:00
- **Git Commit**: `4b230f0d85` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `hla_attention` |
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
| **Peak VRAM** | 2871.27 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module hla_attention --model gpt2 --dataset wikitext2 --rank 32
```
