<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Window $w$ | {{WINDOW_SIZE}} |
| RFF $m$ | {{RFF_FEATURES}} |
| Gate bias | {{GATE_BIAS}} |
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
| Inference time | {{R_TIME}} s |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Mean gate | {{G_GATE}} |
| Peak VRAM | {{G_VRAM}} MB |
| Inference time | {{G_TIME}} s |

## C.4 Per-Layer Gate (DistilBERT)

| Layer | Mean gate |
|---|---|
| 0 | {{D_L0_GATE}} |
| 1 | {{D_L1_GATE}} |
| 2 | {{D_L2_GATE}} |
| 3 | {{D_L3_GATE}} |
| 4 | {{D_L4_GATE}} |
| 5 | {{D_L5_GATE}} |

## C.5 Window Size Ablation

| $w$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | {{D_W16}} | {{R_W16}} | {{G_W16}} |
| 32 | {{D_W32}} | {{R_W32}} | {{G_W32}} |
| 64 | {{D_W64}} | {{R_W64}} | {{G_W64}} |
| 128 | {{D_W128}} | {{R_W128}} | {{G_W128}} |

## C.6 RFF Features Ablation

| $m$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | {{D_M32}} | {{R_M32}} | {{G_M32}} |
| 64 | {{D_M64}} | {{R_M64}} | {{G_M64}} |
| 128 | {{D_M128}} | {{R_M128}} | {{G_M128}} |
| 256 | {{D_M256}} | {{R_M256}} | {{G_M256}} |

## C.7 Gate Bias Ablation

| Bias | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | {{D_B0}} | {{R_B0}} | {{G_B0}} |
| 1.0 | {{D_B1}} | {{R_B1}} | {{G_B1}} |
| 1.8 | {{D_B18}} | {{R_B18}} | {{G_B18}} |
| 3.0 | {{D_B3}} | {{R_B3}} | {{G_B3}} |
| 5.0 | {{D_B5}} | {{R_B5}} | {{G_B5}} |

## C.8 LoRA Rank Ablation

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_LR4}} | {{R_LR4}} | {{G_LR4}} |
| 8 | {{D_LR8}} | {{R_LR8}} | {{G_LR8}} |
| 16 | {{D_LR16}} | {{R_LR16}} | {{G_LR16}} |
| 32 | {{D_LR32}} | {{R_LR32}} | {{G_LR32}} |

## C.9 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## C.10 Raw JSON

```json
{
  "nha": {
    "distilbert": {
      "window_size": {{WINDOW_SIZE}},
      "rff_features": {{RFF_FEATURES}},
      "accuracy_pct": {{D_ACC}},
      "mean_gate": {{D_GATE}},
      "peak_vram_mb": {{D_VRAM}}
    },
    "roberta": {...},
    "gpt2": {...}
  }
}
```

## C.11 Reproduction Commands

```bash
# Main
python run_all_4_nha.py --window_size 64 --rff_features 128 --gate_init_bias 3.0

# Window sweep
for w in 16 32 64 128; do
    python run_all_4_nha.py --window_size $w
done
```

---