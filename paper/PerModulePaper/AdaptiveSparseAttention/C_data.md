<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Sparsity (learned mean) | {{D_SP}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| $\rho_\text{min}$ | {{D_RHO_MIN}} |
| $\rho_\text{max}$ | {{D_RHO_MAX}} |
| $\rho_\text{std}$ | {{D_RHO_STD}} |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Sparsity | {{R_SP}} |
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Peak VRAM | {{R_VRAM}} MB |
| Inference time | {{R_TIME}} s |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Sparsity | {{G_SP}} |
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Peak VRAM | {{G_VRAM}} MB |
| Inference time | {{G_TIME}} s |

## C.4 Sparsity Ablation

| $\rho$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.1 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 0.3 | {{D_S30}} | {{R_S30}} | {{G_S30}} |
| 0.5 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 0.7 | {{D_S70}} | {{R_S70}} | {{G_S70}} |
| 0.9 | {{D_S90}} | {{R_S90}} | {{G_S90}} |

## C.5 Temperature Ablation

| $T$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.01 | {{D_T01}} | {{R_T01}} | {{G_T01}} |
| 0.1 | {{D_T1}} | {{R_T1}} | {{G_T1}} |
| 0.5 | {{D_T5}} | {{R_T5}} | {{G_T5}} |
| 1.0 | {{D_T10}} | {{R_T10}} | {{G_T10}} |

## C.6 LoRA Rank Ablation

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_LR4}} | {{R_LR4}} | {{G_LR4}} |
| 8 | {{D_LR8}} | {{R_LR8}} | {{G_LR8}} |
| 16 | {{D_LR16}} | {{R_LR16}} | {{G_LR16}} |
| 32 | {{D_LR32}} | {{R_LR32}} | {{G_LR32}} |

## C.7 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10_R}} | {{R_S10_R}} | {{G_S10_R}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50_R}} | {{R_S50_R}} | {{G_S50_R}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## C.8 Head Budget Distribution

### DistilBERT
| Head | Layer 0 | Layer 3 | Layer 5 |
|---|---|---|---|
| 0 | {{D_H0_L0}} | {{D_H0_L3}} | {{D_H0_L5}} |
| 1 | {{D_H1_L0}} | {{D_H1_L3}} | {{D_H1_L5}} |
| ... | ... | ... | ... |
| 11 | {{D_H11_L0}} | {{D_H11_L3}} | {{D_H11_L5}} |

## C.9 Training Loss Curves

### DistilBERT
```
Step  Loss  Sparsity
1     {{D_S1}}  {{D_SP1}}
5     {{D_S5}}  {{D_SP5}}
10    {{D_S10}} {{D_SP10}}
...
50    {{D_S50}} {{D_SP50}}
```

## C.10 Raw JSON

```json
{
  "adaptive_sparse": {
    "distilbert": {
      "accuracy_pct": {{D_ACC}},
      "learned_sparsity": {{D_SP}},
      "peak_vram_mb": {{D_VRAM}},
      "inference_time_s": {{D_TIME}}
    },
    "roberta": {...},
    "gpt2": {...}
  }
}
```

## C.11 Reproduction Commands

```bash
# Main
python run_all_4_adaptive_sparse.py --sparsity_ratio 0.5

# Sparsity sweep
for rho in 0.1 0.3 0.5 0.7 0.9; do
    python run_all_4_adaptive_sparse.py --sparsity_ratio $rho
done
```
```

---