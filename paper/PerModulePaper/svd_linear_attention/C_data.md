<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS}} |
| Ratio | {{D_RATIO}}× |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Mode | {{D_MODE}} |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS}} |
| Ratio | {{R_RATIO}}× |
| Accuracy | {{R_ACC}}% |
| ... | ... |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS}} |
| Ratio | {{G_RATIO}}× |
| Accuracy | {{G_ACC}}% |
| ... | ... |

## C.4 Rank Ablation

| $r$ | DistilBERT | Ratio | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 8 | {{D_R8}} | 24× | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | 16× | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | 12× | {{R_R32}} | {{G_R32}} |
| 64 | {{D_R64}} | 8× | {{R_R64}} | {{G_R64}} |
| 128 | {{D_R128}} | 4× | {{R_R128}} | {{G_R128}} |

## C.5 Mode Ablation

| Mode | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| `exact` | {{D_EXACT}} | {{R_EXACT}} | {{G_EXACT}} |
| `linear` | {{D_LINEAR}} | {{R_LINEAR}} | {{G_LINEAR}} |
| `decay_linear` | {{D_DECAY}} | {{R_DECAY}} | {{G_DECAY}} |

## C.6 Decay Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| No decay | {{D_NODECAY}} | {{R_NODECAY}} | {{G_NODECAY}} |
| Fixed decay (γ=0.99) | {{D_FIXED}} | {{R_FIXED}} | {{G_FIXED}} |
| Learned decay | {{D_DECAY}} | {{R_DECAY}} | {{G_DECAY}} |

## C.7 Scaling with Sequence Length

| $N$ | Baseline | SVD-Lin | Speedup |
|---|---|---|---|
| 128 | 4.2 s | {{T128}} | {{S128}}× |
| 256 | 16.8 s | {{T256}} | {{S256}}× |
| 512 | 67 s | {{T512}} | {{S512}}× |
| 1024 | 268 s | {{T1024}} | {{S1024}}× |

## C.8 Training Loss Curves

### DistilBERT
```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
...
50    {{D_S50}}
```

## C.9 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": {{D_PARAMS}},
    "ratio": {{D_RATIO}},
    "accuracy_pct": {{D_ACC}},
    "mode": "{{D_MODE}}",
    "rank": {{RANK}}
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.10 Reproduction Commands

```bash
# Full benchmark
python run_all_4_svd_linear_attention.py --models distilbert,roberta,gpt2

# Rank sweep
for r in 8 16 32 64 128; do
    python run_all_4_svd_linear_attention.py --rank $r --output_suffix r$r
done

# Decay sweep
python run_all_4_svd_linear_attention.py --use_decay_gating false --output_suffix nodecay
python run_all_4_svd_linear_attention.py --use_decay_gating true  --output_suffix decay
```
```

---