<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Landmarks $m$ | {{RANK}} |
| Pinv iters $K$ | {{PINV_ITERS}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Attention memory | {{D_ATTN_MEM}} |
| Inference time | {{D_TIME}} s |
| Peak VRAM | {{D_VRAM}} MB |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Attention memory | {{R_ATTN_MEM}} |
| Inference time | {{R_TIME}} s |
| Peak VRAM | {{R_VRAM}} MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Attention memory | {{G_ATTN_MEM}} |
| Inference time | {{G_TIME}} s |
| Peak VRAM | {{G_VRAM}} MB |

## C.4 Landmark Ablation

| $m$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | {{D_M16}} | {{R_M16}} | {{G_M16}} |
| 32 | {{D_M32}} | {{R_M32}} | {{G_M32}} |
| 64 | {{D_M64}} | {{R_M64}} | {{G_M64}} |
| 128 | {{D_M128}} | {{R_M128}} | {{G_M128}} |
| 256 | {{D_M256}} | {{R_M256}} | {{G_M256}} |

## C.5 Pinv Iterations Ablation

| $K$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | {{D_K0}} | {{R_K0}} | {{G_K0}} |
| 2 | {{D_K2}} | {{R_K2}} | {{G_K2}} |
| 4 | {{D_K4}} | {{R_K4}} | {{G_K4}} |
| 6 | {{D_K6}} | {{R_K6}} | {{G_K6}} |
| 8 | {{D_K8}} | {{R_K8}} | {{G_K8}} |

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

## C.8 Training Loss Curves

### DistilBERT
```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
10    {{D_S10}}
...
50    {{D_S50}}
```

### RoBERTa
```
Step  Loss
1     {{R_S1}}
...
50    {{R_S50}}
```

### GPT-2
```
Step  Loss
1     {{G_S1}}
...
50    {{G_S50}}
```

## C.9 Raw JSON

```json
{
  "primal": {
    "distilbert": {
      "num_landmarks": {{RANK}},
      "pinv_iters": {{PINV_ITERS}},
      "accuracy_pct": {{D_ACC}},
      "attention_mem_bytes": {{D_ATTN_MEM_BYTES}},
      "peak_vram_mb": {{D_VRAM}},
      "inference_time_s": {{D_TIME}}
    },
    "roberta": {...},
    "gpt2": {...}
  }
}
```

## C.10 Reproduction Commands

```bash
# Main
python run_all_4_primal.py --rank 64 --pinv_iters 4

# Rank sweep
for m in 16 32 64 128 256; do
    python run_all_4_primal.py --rank $m --output_suffix m$m
done
```

---