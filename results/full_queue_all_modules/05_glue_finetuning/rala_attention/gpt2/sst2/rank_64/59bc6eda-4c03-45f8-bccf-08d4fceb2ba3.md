# Benchmark Run Summary: rala_attention

- **Run ID**: `59bc6eda-4c03-45f8-bccf-08d4fceb2ba3`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-14T05:44:33Z
- **Git Commit**: `32e519631c` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `rala_attention` |
| **Model** | `gpt2` |
| **Dataset** | `sst2` |
| **Rank** | `64` |
| **Training Type** | `finetune` |
| **Batch Size** | `2` |

## Primary Metrics
| Metric | Value |
|---|---|
| **Pre-KD Perplexity** | N/A |
| **Post-KD Perplexity** | N/A |
| **Recovery %** | N/A |
| **KD Gain** | N/A |
| **Accuracy** | 0.5500 |
| **F1 Score** | 0.5263 |
| **Loss** | 1.2485 |
| **Duration (s)** | 22.94s |
| **Peak VRAM** | 885.00634765625 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module rala_attention --model gpt2 --dataset sst2 --rank 64
```
