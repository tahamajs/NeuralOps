# Benchmark Run Summary: qcqa_attention

- **Run ID**: `kd_qcqa_attention_C_20261012_225935`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T23:01:04.980002+00:00
- **Git Commit**: `323f6aa8df` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `qcqa_attention` |
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
| **Peak VRAM** | 5755.56 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module qcqa_attention --model gpt2 --dataset wikitext2 --rank 32
```
