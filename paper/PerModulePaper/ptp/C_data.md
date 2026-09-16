<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Keep ratio | {{KEEP_RATIO}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Kept ratio | {{D_KEPT}}% |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Speedup | {{D_SPEEDUP}}× |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Kept | {{R_KEPT}}% |
| Speedup | {{R_SPEEDUP}}× |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Kept | {{G_KEPT}}% |
| Speedup | {{G_SPEEDUP}}× |

## C.4 Keep Ratio Ablation

| $\rho$ | DistilBERT | RoBERTa | GPT-2 | Speedup |
|---|---|---|---|---|
| 0.5 | {{D_R50}} | {{R_R50}} | {{G_R50}} | 2.0× |
| 0.7 | {{D_R70}} | {{R_R70}} | {{G_R70}} | 1.43× |
| 0.8 | {{D_R80}} | {{R_R80}} | {{G_R80}} | 1.25× |
| 0.9 | {{D_R90}} | {{R_R90}} | {{G_R90}} | 1.11× |
| 0.95 | {{D_R95}} | {{R_R95}} | {{G_R95}} | 1.05× |

## C.5 [CLS] Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With [CLS] | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} |
| Without | {{D_NOCLS}} | {{R_NOCLS}} | {{G_NOCLS}} |

## C.6 Score Network Hidden

| Hidden | DistilBERT | Params |
|---|---|---|
| 32 | {{D_H32}} | ~25K |
| 64 | {{D_H64}} | ~50K |
| 128 | {{D_H128}} | ~99K |

## C.7 LoRA Rank Ablation

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_LR4}} | {{R_LR4}} | {{G_LR4}} |
| 8 | {{D_LR8}} | {{R_LR8}} | {{G_LR8}} |
| 16 | {{D_LR16}} | {{R_LR16}} | {{G_LR16}} |
| 32 | {{D_LR32}} | {{R_LR32}} | {{G_LR32}} |

## C.8 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## C.9 Token Score Analysis

| POS tag | Mean score |
|---|---|
| Noun | {{SCORE_NOUN}} |
| Verb | {{SCORE_VERB}} |
| Adjective | {{SCORE_ADJ}} |
| Adverb | {{SCORE_ADV}} |
| Determiner | {{SCORE_DET}} |
| Preposition | {{SCORE_PREP}} |
| Punctuation | {{SCORE_PUNCT}} |

## C.10 Raw JSON

```json
{
  "ptp": {
    "distilbert": {
      "keep_ratio": {{KEEP_RATIO}},
      "accuracy_pct": {{D_ACC}},
      "speedup": {{D_SPEEDUP}}
    },
    "roberta": {...},
    "gpt2": {...}
  }
}
```

## C.11 Reproduction Commands

```bash
# Main
python run_all_4_ptp.py --keep_ratio 0.8

# Keep ratio sweep
for rho in 0.5 0.7 0.8 0.9 0.95; do
    python run_all_4_ptp.py --keep_ratio $rho
done
```

---