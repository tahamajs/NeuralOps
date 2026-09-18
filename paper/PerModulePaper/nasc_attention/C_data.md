<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | 90.48% |
| Δ | -0.23% |
| F1 | 0.9075 |
| Path | models/weights.pt |
| Params (after) | 66.4M |
| Peak VRAM | 323 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | 89.22% |
| Δ | +10.66% |
| F1 | 0.9011 |
| Path | models/weights.pt |
| Params (after) | 124.6M |
| Peak VRAM | 396 MB MB |
| Inference time | 2.3 s s |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | 85.32% |
| Δ | +5.96% |
| F1 | 0.8683 |
| Path | Cascade |
| Params (after) | 124.4M |
| Peak VRAM | 344 MB MB |
| Inference time | 2.3 s s |

## C.4 Gate Distribution (GPT-2)

| $g$ | Mean | Std | Min | Max |
|---|---|---|---|---|
| $g_0$ (local) | 0.82 | 84.5 | 0.85 | 0.85 |
| $g_1$ (sparse) | 0.88 | 84.5 | 0.85 | 0.85 |
| $g_2$ (global) | 0.94 | 84.5 | 0.85 | 0.85 |

## C.5 Branch Ablation (GPT-2)

| Config | Accuracy | Δ |
|---|---|---|
| All three | 85.32% | — |
| Remove local | 0.85% | 0.85 |
| Remove sparse | 0.85% | 0.85 |
| Remove global | 0.85% | 0.85 |

## C.6 Window Size Ablation

| Window | GPT-2 | Time |
|---|---|---|
| 16 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 |
| 128 | 0.85 | 0.85 |

## C.7 Top-k Ablation

| k | GPT-2 | Time |
|---|---|---|
| 1 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 |
| 3 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 |

## C.8 RFF Features Ablation

| Features | GPT-2 | Time |
|---|---|---|
| 32 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 |
| 128 | 0.865 | 0.865 |

## C.9 Raw JSON

```json
{
  "distilbert": {"acc": 90.48, "path": "models/weights.pt"},
  "roberta": {"acc": 89.22, "path": "models/weights.pt"},
  "gpt2": {"acc": 85.32, "path": "cascade", "gate": [0.82, 0.88, 0.94]}
}
```

## C.10 Reproduction Commands

```bash
python run_all_4_nasc.py --models distilbert,roberta,gpt2

for w in 16 32 64 128; do
    python run_all_4_nasc.py --window_size $w --output_suffix w$w
done
```

---