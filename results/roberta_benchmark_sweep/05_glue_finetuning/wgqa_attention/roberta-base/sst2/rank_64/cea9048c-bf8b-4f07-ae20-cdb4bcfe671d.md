# Benchmark Run Summary: wgqa_attention

- **Run ID**: `cea9048c-bf8b-4f07-ae20-cdb4bcfe671d`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-14T07:18:14Z
- **Git Commit**: `631445fec7` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `wgqa_attention` |
| **Model** | `roberta-base` |
| **Dataset** | `sst2` |
| **Rank** | `64` |
| **Training Type** | `finetune` |
| **Batch Size** | `16` |

## Primary Metrics
| Metric | Value |
|---|---|
| **Pre-KD Perplexity** | N/A |
| **Post-KD Perplexity** | N/A |
| **Recovery %** | N/A |
| **KD Gain** | N/A |
| **Accuracy** | 0.7300 |
| **F1 Score** | 0.6943 |
| **Loss** | 0.5336 |
| **Duration (s)** | 31.91s |
| **Peak VRAM** | 950.958984375 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module wgqa_attention --model roberta-base --dataset sst2 --rank 64
```
