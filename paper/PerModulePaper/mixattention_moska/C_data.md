<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | {{D_ACC}}% |
| Δ | {{D_DELTA}}% |
| F1 | {{D_F1}} |
| Params (after) | {{D_PARAMS}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Path | Pruned |
| Keep ratio | {{KEEP_RATIO}} |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | {{R_ACC}}% |
| Δ | {{R_DELTA}}% |
| F1 | {{R_F1}} |
| Params (after) | {{R_PARAMS}} |
| Peak VRAM | {{R_VRAM}} MB |
| Inference time | {{R_TIME}} s |
| Path | Pruned |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | {{G_ACC}}% |
| Δ | {{G_DELTA}}% |
| F1 | {{G_F1}} |
| Params (after) | {{G_PARAMS}} |
| Peak VRAM | {{G_VRAM}} MB |
| Inference time | {{G_TIME}} s |
| Path | **No prune (causal)** |

## C.4 Keep Ratio Ablation

| Ratio | DistilBERT | RoBERTa |
|---|---|---|
| 0.25 | {{D_K25}} | {{R_K25}} |
| 0.50 | {{D_K50}} | {{R_K50}} |
| 0.75 | {{D_K75}} | {{R_K75}} |
| 1.00 | {{D_K100}} | {{R_K100}} |

## C.5 Importance Method

| Method | DistilBERT | RoBERTa |
|---|---|---|
| `var` | {{D_VAR}} | {{R_VAR}} |
| `l2` | {{D_L2}} | {{R_L2}} |

## C.6 CLS Protection

| Setting | DistilBERT | RoBERTa |
|---|---|---|
| Protect | {{D_PROTECT_ON}} | {{R_PROTECT_ON}} |
| No protect | {{D_PROTECT_OFF}} | {{R_PROTECT_OFF}} |

## C.7 Raw JSON

```json
{
  "distilbert": {"acc": {{D_ACC}}, "path": "pruned"},
  "roberta": {"acc": {{R_ACC}}, "path": "pruned"},
  "gpt2": {"acc": {{G_ACC}}, "path": "no-prune"}
}
```

## C.8 Reproduction Commands

```bash
python run_all_4_fasa.py --models distilbert,roberta,gpt2

for r in 0.25 0.5 0.75 1.0; do
    python run_all_4_fasa.py --fasa_keep_ratio $r --output_suffix k$r
done
```
```

---