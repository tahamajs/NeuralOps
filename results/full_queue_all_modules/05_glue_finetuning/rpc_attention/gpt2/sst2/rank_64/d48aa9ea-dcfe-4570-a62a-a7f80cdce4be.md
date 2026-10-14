# Benchmark Run Summary: rpc_attention

- **Run ID**: `d48aa9ea-dcfe-4570-a62a-a7f80cdce4be`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-14T05:47:29Z
- **Git Commit**: `32e519631c` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `rpc_attention` |
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
| **Accuracy** | 0.4500 |
| **F1 Score** | 0.2667 |
| **Loss** | 0.8492 |
| **Duration (s)** | 17.54s |
| **Peak VRAM** | 798.36962890625 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module rpc_attention --model gpt2 --dataset sst2 --rank 64
```
