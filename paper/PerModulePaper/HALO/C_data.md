<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Mean gate | {{D_GATE}} |
| Gate min | {{D_GATE_MIN}} |
| Gate max | {{D_GATE_MAX}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Mean gate | {{R_GATE}} |
| Peak VRAM | {{R_VRAM}} MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Mean gate | {{G_GATE}} |
| Peak VRAM | {{G_VRAM}} MB |

## C.4 Per-Layer Gate (DistilBERT)

| Layer | Mean gate |
|---|---|
| 0 | {{D_L0_GATE}} |
| 1 | {{D_L1_GATE}} |
| 2 | {{D_L2_GATE}} |
| 3 | {{D_L3_GATE}} |
| 4 | {{D_L4_GATE}} |
| 5 | {{D_L5_GATE}} |

## C.5 Gate Bias Ablation

| $b_h$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | {{D_B0}} | {{R_B0}} | {{G_B0}} |
| 0.5 | {{D_B05}} | {{R_B05}} | {{G_B05}} |
| 1.0 | {{D_B1}} | {{R_B1}} | {{G_B1}} |
| 1.8 | {{D_B18}} | {{R_B18}} | {{G_B18}} |
| 3.0 | {{D_B3}} | {{R_B3}} | {{G_B3}} |

## C.6 Kernel Ablation

| Kernel | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| ELU+1 | {{D_ELU}} | {{R_ELU}} | {{G_ELU}} |
| ReLU+1 | {{D_RELU}} | {{R_RELU}} | {{G_RELU}} |
| Softplus | {{D_SOFTPLUS}} | {{R_SOFTPLUS}} | {{G_SOFTPLUS}} |
| Identity | {{D_IDENT}} | {{R_IDENT}} | {{G_IDENT}} |

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

## C.9 Raw JSON

```json
{
  "halo_regla": {
    "distilbert": {
      "accuracy_pct": {{D_ACC}},
      "mean_gate": {{D_GATE}},
      "peak_vram_mb": {{D_VRAM}}
    },
    "roberta": {...},
    "gpt2": {...}
  }
}
```

## C.10 Reproduction Commands

```bash
# Main
python run_all_4_halo_regla.py --gate_init_bias 1.8

# Bias sweep
for b in 0 0.5 1.0 1.8 3.0; do
    python run_all_4_halo_regla.py --gate_init_bias $b
done
```
