# Benchmark Run Summary: qcqa_attention

- **Run ID**: `kd_qcqa_attention_C_20261013_080551`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-13T08:24:45.670266+00:00
- **Git Commit**: `6c7a3c0c97` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `qcqa_attention` |
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
| **Peak VRAM** | 2888.41 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module qcqa_attention --model gpt2 --dataset wikitext2 --rank 32
```
