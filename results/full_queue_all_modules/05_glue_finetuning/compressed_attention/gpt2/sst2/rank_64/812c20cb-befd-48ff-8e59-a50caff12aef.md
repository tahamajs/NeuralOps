# Benchmark Run Summary: compressed_attention

- **Run ID**: `812c20cb-befd-48ff-8e59-a50caff12aef`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-14T05:16:55Z
- **Git Commit**: `32e519631c` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `compressed_attention` |
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
| **F1 Score** | 0.5926 |
| **Loss** | 0.8399 |
| **Duration (s)** | 15.52s |
| **Peak VRAM** | 787.13134765625 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module compressed_attention --model gpt2 --dataset sst2 --rank 64
```
