# Benchmark Run Summary: gla_attention

- **Run ID**: `kd_gla_attention_C_20261013_094812`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-13T09:48:43.903623+00:00
- **Git Commit**: `89adc714db` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `gla_attention` |
| **Model** | `gpt2` |
| **Dataset** | `wikitext2` |
| **Rank** | `16` |
| **Training Type** | `finetune` |
| **Batch Size** | `1` |

## Primary Metrics
| Metric | Value |
|---|---|
| **Pre-KD Perplexity** | 3946.45 |
| **Post-KD Perplexity** | 2172.47 |
| **Recovery %** | 45.37% |
| **KD Gain** | -1159.06 |
| **Accuracy** | N/A |
| **F1 Score** | N/A |
| **Loss** | N/A |
| **Duration (s)** | 11.69s |
| **Peak VRAM** | 2110.56 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module gla_attention --model gpt2 --dataset wikitext2 --rank 16
```
