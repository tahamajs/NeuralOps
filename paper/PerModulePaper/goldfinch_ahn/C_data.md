<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | {{D_ACC}}% |
| Δ | {{D_DELTA}}% |
| F1 | {{D_F1}} |
| Params (before) | 66,955,010 |
| Params (after) | {{D_PARAMS}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | {{R_ACC}}% |
| Δ | {{R_DELTA}}% |
| F1 | {{R_F1}} |
| Params (after) | {{R_PARAMS}} |
| Peak VRAM | {{R_VRAM}} MB |
| Inference time | {{R_TIME}} s |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | {{G_ACC}}% |
| Δ | {{G_DELTA}}% |
| F1 | {{G_F1}} |
| Params (after) | {{G_PARAMS}} |
| Peak VRAM | {{G_VRAM}} MB |
| Inference time | {{G_TIME}} s |

## C.4 Window Size Ablation

| Window | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | {{D_W16}} | {{R_W16}} | {{G_W16}} |
| 32 | {{D_W32}} | {{R_W32}} | {{G_W32}} |
| 64 | {{D_W64}} | {{R_W64}} | {{G_W64}} |
| 128 | {{D_W128}} | {{R_W128}} | {{G_W128}} |
| ∞ (full) | {{D_WINF}} | {{R_WINF}} | {{G_WINF}} |

## C.5 Gate Bias Ablation

| Bias | Initial g | DistilBERT |
|---|---|---|
| 0.0 | 0.50 | {{D_BIAS0}} |
| 1.0 | 0.73 | {{D_BIAS1}} |
| 1.8 | 0.86 | {{D_BIAS18}} |
| 3.0 | 0.95 | {{D_BIAS3}} |

## C.6 Preservation Gate Ablation

| Init | Initial α | DistilBERT |
|---|---|---|
| 0.0 | 0.50 | {{D_AG0}} |
| 1.0 | 0.73 | {{D_AG1}} |
| 2.0 | 0.88 | {{D_AG2}} |
| 5.0 | 0.99 | {{D_AG5}} |

## C.7 Remove Long-Memory Branch

| Config | DistilBERT |
|---|---|
| Full | {{D_ACC}}% |
| Without long-memory | {{D_NOLONG}}% |

## C.8 Raw JSON

```json
{
  "distilbert": {"acc": {{D_ACC}}, "params": {{D_PARAMS}}},
  "roberta": {"acc": {{R_ACC}}, "params": {{R_PARAMS}}},
  "gpt2": {"acc": {{G_ACC}}, "params": {{G_PARAMS}}}
}
```

## C.9 Reproduction Commands

```bash
python run_all_4_goldfinch_ahn.py --models distilbert,roberta,gpt2

for W in 16 32 64 128; do
    python run_all_4_goldfinch_ahn.py --window_size $W --output_suffix w$W
done
```
```

---