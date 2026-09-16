<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     DEPENDENCIES: 07_results.md
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

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | {{R_ACC}}% |
| Δ | {{R_DELTA}}% |
| F1 | {{R_F1}} |
| Params (after) | {{R_PARAMS}} |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | {{G_ACC}}% |
| Δ | {{G_DELTA}}% |
| F1 | {{G_F1}} |
| Params (after) | {{G_PARAMS}} |

## C.4 Feature Ablation

| $F$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | {{D_F32}} | {{R_F32}} | {{G_F32}} |
| 64 | {{D_F64}} | {{R_F64}} | {{G_F64}} |
| 128 | {{D_F128}} | {{R_F128}} | {{G_F128}} |
| 256 | {{D_F256}} | {{R_F256}} | {{G_F256}} |

## C.5 Kernel Ablation

| Kernel | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| `favor_plus` | {{D_FAVOR}} | {{R_FAVOR}} | {{G_FAVOR}} |
| `elu` | {{D_ELU}} | {{R_ELU}} | {{G_ELU}} |
| `relu` | {{D_RELU}} | {{R_RELU}} | {{G_RELU}} |

## C.6 Raw JSON

```json
{
  "distilbert": {"acc": {{D_ACC}}, "params": {{D_PARAMS}}},
  "roberta": {"acc": {{R_ACC}}, "params": {{R_PARAMS}}},
  "gpt2": {"acc": {{G_ACC}}, "params": {{G_PARAMS}}}
}
```

## C.7 Reproduction Commands

```bash
python run_all_4_performer.py --models distilbert,roberta,gpt2

for F in 32 64 128 256; do
    python run_all_4_performer.py --num_features $F --output_suffix f$F
done
```
```

---