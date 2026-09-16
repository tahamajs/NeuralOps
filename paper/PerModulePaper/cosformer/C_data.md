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
| Precision | {{D_PREC}} |
| Recall | {{D_REC}} |
| Loss | {{D_LOSS}} |
| Params (before) | 66,955,010 |
| Params (after) | {{D_PARAMS}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Path | Encoder |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | {{R_ACC}}% |
| Δ | {{R_DELTA}}% |
| F1 | {{R_F1}} |
| Params (before) | 124,647,170 |
| Params (after) | {{R_PARAMS}} |
| Peak VRAM | {{R_VRAM}} MB |
| Inference time | {{R_TIME}} s |
| Path | Encoder (nested) |
| Status | {{R_STATUS}} |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | {{G_ACC}}% |
| Δ | {{G_DELTA}}% |
| F1 | {{G_F1}} |
| Params (before) | 124,441,344 |
| Params (after) | {{G_PARAMS}} |
| Peak VRAM | {{G_VRAM}} MB |
| Inference time | {{G_TIME}} s |
| Path | Decoder |

## C.4 Chunk Size Ablation

| chunk_size | GPT-2 time | GPT-2 acc |
|---|---|---|
| 16 | {{G_CHUNK16_TIME}} | {{G_CHUNK16_ACC}} |
| 32 | {{G_CHUNK32_TIME}} | {{G_CHUNK32_ACC}} |
| 64 | {{G_CHUNK64_TIME}} | {{G_CHUNK64_ACC}} |
| 128 | {{G_CHUNK128_TIME}} | {{G_CHUNK128_ACC}} |

## C.5 Activation Ablation

| Act | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| `elu+1` | {{D_ELU}} | {{R_ELU}} | {{G_ELU}} |
| `softplus` | {{D_SOFTPLUS}} | {{R_SOFTPLUS}} | {{G_SOFTPLUS}} |

## C.6 Training Loss

### DistilBERT

```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
10    {{D_S10}}
...
50    {{D_S50}}
```

## C.7 Raw JSON

```json
{
  "distilbert": {"acc": {{D_ACC}}, "path": "encoder"},
  "roberta": {"acc": {{R_ACC}}, "path": "encoder_nested"},
  "gpt2": {"acc": {{G_ACC}}, "path": "decoder"}
}
```

## C.8 Reproduction Commands

```bash
python run_all_4_cosformer.py --models distilbert,roberta,gpt2

for cs in 16 32 64 128; do
    python run_all_4_cosformer.py --chunk_size $cs --output_suffix cs$cs
done
```

---