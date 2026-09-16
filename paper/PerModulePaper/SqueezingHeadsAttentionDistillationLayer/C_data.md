<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| LoRA rank | {{LORA_RANK}} |
| Head-dim reduction | {{HEAD_DIM_REDUCTION}}× |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Trainable params | {{D_LORA}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Trainable params | {{R_LORA}} |
| Peak VRAM | {{R_VRAM}} MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Trainable params | {{G_LORA}} |
| Peak VRAM | {{G_VRAM}} MB |

## C.4 LoRA Rank Ablation

| $r_L$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_LR4}} | {{R_LR4}} | {{G_LR4}} |
| 8 | {{D_LR8}} | {{R_LR8}} | {{G_LR8}} |
| 16 | {{D_LR16}} | {{R_LR16}} | {{G_LR16}} |
| 32 | {{D_LR32}} | {{R_LR32}} | {{G_LR32}} |

## C.5 Head-Dim Reduction Ablation

| $s$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1.0 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 1.5 | {{D_S15}} | {{R_S15}} | {{G_S15}} |
| 2.0 | {{D_S20}} | {{R_S20}} | {{G_S20}} |
| 2.5 | {{D_S25}} | {{R_S25}} | {{G_S25}} |

## C.6 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10_STEP}} | {{R_S10_STEP}} | {{G_S10_STEP}} |
| 25 | {{D_S25_STEP}} | {{R_S25_STEP}} | {{G_S25_STEP}} |
| 50 | {{D_S50_STEP}} | {{R_S50_STEP}} | {{G_S50_STEP}} |
| 100 | {{D_S100_STEP}} | {{R_S100_STEP}} | {{G_S100_STEP}} |

## C.7 Per-Layer Analysis

| Layer | DistilBERT drop | RoBERTa drop |
|---|---|---|
| 0 | {{D_L0_DROP}} | {{R_L0_DROP}} |
| 1 | {{D_L1_DROP}} | {{R_L1_DROP}} |
| ... | ... | ... |

## C.8 Raw JSON

```json
{
  "squeezing_heads": {
    "distilbert": {
      "lora_rank": {{LORA_RANK}},
      "head_dim_reduction": {{HEAD_DIM_REDUCTION}},
      "accuracy_pct": {{D_ACC}},
      "trainable_params": {{D_LORA}}
    },
    "roberta": {...},
    "gpt2": {...}
  }
}
```

## C.9 Reproduction Commands

```bash
# Main
python run_all_4_squeezing_heads.py --lora_rank 16

# Rank sweep
for r in 4 8 16 32; do
    python run_all_4_squeezing_heads.py --lora_rank $r
done

# Head-dim reduction sweep
for s in 1.0 1.5 2.0 2.5; do
    python run_all_4_squeezing_heads.py --head_dim_reduction $s
done
```

---