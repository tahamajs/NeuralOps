<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT — Path A (MHA)

| Metric | Value |
|---|---|
| Accuracy | {{D_MHA_ACC}}% |
| Δ | {{D_MHA_DELTA}}% |
| Params (after) | {{D_MHA_PARAMS}} |
| Peak VRAM | {{D_MHA_VRAM}} MB |
| Per-step time | {{D_MHA_TIME}} s |

## C.2 DistilBERT — Path B (DeltaNet)

| Metric | Value |
|---|---|
| Accuracy | {{D_ACC}}% |
| Δ | {{D_DELTA}}% |
| Params (after) | {{D_PARAMS}} |
| Peak VRAM | {{D_VRAM}} MB |
| Per-step time | {{D_STEP_TIME}} s |

## C.3 RoBERTa Status

| Metric | Path A | Path B |
|---|---|---|
| Status | {{R_MHA_STATUS}} | OOM |
| Error | {{R_MHA_ERROR}} | CUDA OOM |

## C.4 GPT-2 Status

| Metric | Path A | Path B |
|---|---|---|
| Status | {{G_MHA_STATUS}} | OOM |
| Error | {{G_MHA_ERROR}} | CUDA OOM |

## C.5 Weight Transfer Verification

| Model | Q/K/V shape match? | Out_proj match? |
|---|---|---|
| DistilBERT | {{D_QKV_OK}} | {{D_O_OK}} |
| RoBERTa | {{R_QKV_OK}} | {{R_O_OK}} |
| GPT-2 | {{G_QKV_OK}} | {{G_O_OK}} |

## C.6 LayerNorm After Recovery (DistilBERT)

| Metric | Value |
|---|---|
| γ mean | {{D_GAMMA_MEAN}} |
| γ std | {{D_GAMMA_STD}} |
| β mean | {{D_BETA_MEAN}} |
| β std | {{D_BETA_STD}} |

## C.7 Raw JSON

```json
{
  "distilbert_mha": {
    "accuracy": {{D_MHA_ACC}},
    "params": {{D_MHA_PARAMS}},
    "vram_mb": {{D_MHA_VRAM}}
  },
  "distilbert_deltanet": {
    "accuracy": {{D_ACC}},
    "params": {{D_PARAMS}},
    "vram_mb": {{D_VRAM}}
  }
}
```

## C.8 Reproduction Commands

```bash
# Path A
python run_all_4_gated_deltanet_block.py --models distilbert --use_full_attention true

# Path B
python run_all_4_gated_deltanet_block.py --models distilbert --use_full_attention false
```

---