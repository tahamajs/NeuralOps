<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Max span | {{MAX_SPAN}} |
| Ramp size | {{RAMP_SIZE}} |
| λ | {{LAMBDA}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Mean span | {{D_SPAN}} |
| Min/Max span | {{D_SPAN_MIN}} / {{D_SPAN_MAX}} |
| Std span | {{D_SPAN_STD}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Mean span | {{R_SPAN}} |
| Reduction | {{R_SPAN_REDUCTION}}% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Mean span | {{G_SPAN}} |
| Reduction | {{G_SPAN_REDUCTION}}% |

## C.4 Per-Layer Span (DistilBERT)

| Layer | Mean span |
|---|---|
| 0 | {{D_L0_SPAN}} |
| 1 | {{D_L1_SPAN}} |
| 2 | {{D_L2_SPAN}} |
| 3 | {{D_L3_SPAN}} |
| 4 | {{D_L4_SPAN}} |
| 5 | {{D_L5_SPAN}} |

## C.5 Max Span Ablation

| $S_\text{max}$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | {{D_MAX32}} | {{R_MAX32}} | {{G_MAX32}} |
| 64 | {{D_MAX64}} | {{R_MAX64}} | {{G_MAX64}} |
| 128 | {{D_MAX128}} | {{R_MAX128}} | {{G_MAX128}} |
| 256 | {{D_MAX256}} | {{R_MAX256}} | {{G_MAX256}} |

## C.6 Ramp Size Ablation

| $R$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | {{D_R1}} | {{R_R1}} | {{G_R1}} |
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} |
| 64 | {{D_R64}} | {{R_R64}} | {{G_R64}} |

## C.7 L1 Penalty Ablation

| $\lambda$ | DistilBERT acc | Mean span |
|---|---|---|
| 0.0 | {{D_L0}} | {{D_SPAN0}} |
| 0.001 | {{D_L001}} | {{D_SPAN001}} |
| 0.01 | {{D_L01}} | {{D_SPAN01}} |
| 0.1 | {{D_L1}} | {{D_SPAN1}} |

## C.8 Head Span Distribution (DistilBERT Layer 3)

| Head | Span |
|---|---|
| 0 | {{D_L3_H0}} |
| 1 | {{D_L3_H1}} |
| ... | ... |
| 11 | {{D_L3_H11}} |

## C.9 Span Evolution

### DistilBERT
```
Step  Mean span
0     {{D_STEP0}}
5     {{D_STEP5}}
10    {{D_STEP10}}
15    {{D_STEP15}}
20    {{D_STEP20}}
...
50    {{D_STEP50}}
```

## C.10 Raw JSON

```json
{
  "adaptive_span": {
    "distilbert": {
      "max_span": {{MAX_SPAN}},
      "learned_mean_span": {{D_SPAN}},
      "accuracy_pct": {{D_ACC}},
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
python run_all_4_adaptive_span.py --max_span 128 --ramp_size 32

# Max span sweep
for S in 32 64 128 256; do
    python run_all_4_adaptive_span.py --max_span $S
done
```

---