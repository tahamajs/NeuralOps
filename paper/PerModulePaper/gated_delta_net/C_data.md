<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | {{D_ACC}}% |
| Δ | {{D_DELTA}}% |
| F1 | {{D_F1}} |
| Params (before) | 66,955,010 |
| Params (after) | {{D_PARAMS}} |
| Δ params | +{{D_PARAM_ABS}} |
| Δ% | +{{D_PARAM_PCT}}% |
| Peak VRAM | {{D_VRAM}} MB |
| Per-step time | {{D_STEP_TIME}} s |
| Total time (5 steps) | {{D_TOTAL_TIME}} s |
| Chunk size | {{CHUNK_SIZE}} |
| Gradient checkpointing | {{USE_GC}} |

## C.2 RoBERTa Status

| Metric | Value |
|---|---|
| Status | {{R_STATUS}} |
| Error | {{R_ERROR}} |
| VRAM at OOM | {{R_OOM_VRAM}} MB |
| Recovery step at OOM | {{R_OOM_STEP}} |

## C.3 GPT-2 Status

| Metric | Value |
|---|---|
| Status | {{G_STATUS}} |
| Error | {{G_ERROR}} |
| VRAM at OOM | {{G_OOM_VRAM}} MB |

## C.4 Timing Breakdown (DistilBERT)

| Phase | Time (ms) |
|---|---|
| Q/K/V projections | 30 |
| Z projection | 10 |
| RoPE | 10 |
| Gating | 20 |
| **Recurrence** | **{{D_RECURRENCE_MS}}** |
| SiLU | 10 |
| Output projection | 10 |
| **Total** | **{{D_STEP_MS}}** |

## C.5 Chunk Size Ablation

| Chunk size | Per-step time | Peak VRAM |
|---|---|---|
| 4 | {{D_CHUNK4_TIME}} s | {{D_CHUNK4_VRAM}} MB |
| 8 | {{D_CHUNK8_TIME}} s | {{D_CHUNK8_VRAM}} MB |
| 16 | {{D_CHUNK16_TIME}} s | {{D_CHUNK16_VRAM}} MB |
| 32 | {{D_CHUNK32_TIME}} s | {{D_CHUNK32_VRAM}} MB |

## C.6 Gradient Checkpointing Ablation

| Setting | Time | VRAM | Accuracy |
|---|---|---|---|
| True | {{D_GC_TRUE_TIME}} s | {{D_GC_TRUE_VRAM}} MB | {{D_GC_TRUE_ACC}}% |
| False | {{D_GC_FALSE_TIME}} s | {{D_GC_FALSE_VRAM}} MB | {{D_GC_FALSE_ACC}}% |

## C.7 Training Loss (DistilBERT)

```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
10    {{D_S10}}
15    {{D_S15}}
20    {{D_S20}}
25    {{D_S25}}
30    {{D_S30}}
35    {{D_S35}}
40    {{D_S40}}
45    {{D_S45}}
50    {{D_S50}}
```

## C.8 Raw JSON

```json
{
  "distilbert": {
    "accuracy": {{D_ACC}},
    "status": "{{D_STATUS}}",
    "per_step_s": {{D_STEP_TIME}},
    "peak_vram_mb": {{D_VRAM}},
    "params_after": {{D_PARAMS}}
  },
  "roberta": {
    "status": "{{R_STATUS}}",
    "error": "{{R_ERROR}}"
  },
  "gpt2": {
    "status": "{{G_STATUS}}",
    "error": "{{G_ERROR}}"
  }
}
```

## C.9 Reproduction Commands

```bash
# DistilBERT only
python run_all_4_gated_delta_net.py --models distilbert

# With smaller batch
python run_all_4_gated_delta_net.py \
    --models distilbert \
    --batch_size 4 \
    --recovery_steps 20

# With chunk size sweep
for cs in 4 8 16 32; do
    python run_all_4_gated_delta_net.py --chunk_size $cs --output_suffix cs$cs
done
```

---