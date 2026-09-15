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
| Top-$k$ | {{TOP_K}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Density | {{D_DENSITY}}% |
| Preserved mass | {{D_MASS}}% |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Density | {{R_DENSITY}}% |
| Preserved mass | {{R_MASS}}% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Density | {{G_DENSITY}}% |
| Preserved mass | {{G_MASS}}% |

## C.4 Window Size Ablation

| $w$ | DistilBERT | RoBERTa | GPT-2 | Density |
|---|---|---|---|---|
| 8 | {{D_W8}} | {{R_W8}} | {{G_W8}} | {{DENS8}}% |
| 16 | {{D_W16}} | {{R_W16}} | {{G_W16}} | {{DENS16}}% |
| 32 | {{D_W32}} | {{R_W32}} | {{G_W32}} | {{DENS32}}% |
| 64 | {{D_W64}} | {{R_W64}} | {{G_W64}} | {{DENS64}}% |

## C.5 Top-K Ablation

| $k$ | DistilBERT | RoBERTa | GPT-2 | Density |
|---|---|---|---|---|
| 4 | {{D_K4}} | {{R_K4}} | {{G_K4}} | {{DENSK4}}% |
| 8 | {{D_K8}} | {{R_K8}} | {{G_K8}} | {{DENSK8}}% |
| 16 | {{D_K16}} | {{R_K16}} | {{G_K16}} | {{DENSK16}}% |
| 32 | {{D_K32}} | {{R_K32}} | {{G_K32}} | {{DENSK32}}% |

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
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## C.8 Per-Layer Density (DistilBERT)

| Layer | Density |
|---|---|
| 0 | {{D_L0_DENS}}% |
| 1 | {{D_L1_DENS}}% |
| 2 | {{D_L2_DENS}}% |
| 3 | {{D_L3_DENS}}% |
| 4 | {{D_L4_DENS}}% |
| 5 | {{D_L5_DENS}}% |

## C.9 Raw JSON

```json
{
  "nsa": {
    "distilbert": {
      "window_size": {{WINDOW_SIZE}},
      "top_k": {{TOP_K}},
      "accuracy_pct": {{D_ACC}},
      "density_pct": {{D_DENSITY}},
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
python run_all_4_nsa.py --window_size 32 --top_k 16

# Window sweep
for w in 8 16 32 64; do
    python run_all_4_nsa.py --window_size $w
done
```
