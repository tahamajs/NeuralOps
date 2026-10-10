# Benchmark Run Summary: mixattention_moska_inject

- **Run ID**: `kd_mixattention_moska_inject_C_20261010_065508`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-10T03:30:17.330116+00:00
- **Git Commit**: `3e5c8ecdb3` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `mixattention_moska_inject` |
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
| **Peak VRAM** | 2234.24 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module mixattention_moska_inject --model gpt2 --dataset wikitext2 --rank 32
```
