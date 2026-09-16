<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | {{D_ACC}}% |
| Δ | {{D_DELTA}}% |
| F1 | {{D_F1}} |
| Path | {{D_PATH}} |
| Params (after) | {{D_PARAMS}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | {{R_ACC}}% |
| Δ | {{R_DELTA}}% |
| F1 | {{R_F1}} |
| Path | {{R_PATH}} |
| Params (after) | {{R_PARAMS}} |
| Peak VRAM | {{R_VRAM}} MB |
| Inference time | {{R_TIME}} s |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | {{G_ACC}}% |
| Δ | {{G_DELTA}}% |
| F1 | {{G_F1}} |
| Path | Cascade |
| Params (after) | {{G_PARAMS}} |
| Peak VRAM | {{G_VRAM}} MB |
| Inference time | {{G_TIME}} s |

## C.4 Gate Distribution (GPT-2)

| $g$ | Mean | Std | Min | Max |
|---|---|---|---|---|
| $g_0$ (local) | {{G_G0}} | {{G_G0_STD}} | {{G_G0_MIN}} | {{G_G0_MAX}} |
| $g_1$ (sparse) | {{G_G1}} | {{G_G1_STD}} | {{G_G1_MIN}} | {{G_G1_MAX}} |
| $g_2$ (global) | {{G_G2}} | {{G_G2_STD}} | {{G_G2_MIN}} | {{G_G2_MAX}} |

## C.5 Branch Ablation (GPT-2)

| Config | Accuracy | Δ |
|---|---|---|
| All three | {{G_ACC}}% | — |
| Remove local | {{G_NOLOC}}% | {{G_NOLOC_D}} |
| Remove sparse | {{G_NOSP}}% | {{G_NOSP_D}} |
| Remove global | {{G_NOGLB}}% | {{G_NOGLB_D}} |

## C.6 Window Size Ablation

| Window | GPT-2 | Time |
|---|---|---|
| 16 | {{G_W16}} | {{G_W16_T}} |
| 32 | {{G_W32}} | {{G_W32_T}} |
| 64 | {{G_W64}} | {{G_W64_T}} |
| 128 | {{G_W128}} | {{G_W128_T}} |

## C.7 Top-k Ablation

| k | GPT-2 | Time |
|---|---|---|
| 1 | {{G_K1}} | {{G_K1_T}} |
| 2 | {{G_K2}} | {{G_K2_T}} |
| 3 | {{G_K3}} | {{G_K3_T}} |
| 4 | {{G_K4}} | {{G_K4_T}} |

## C.8 RFF Features Ablation

| Features | GPT-2 | Time |
|---|---|---|
| 32 | {{G_F32}} | {{G_F32_T}} |
| 64 | {{G_F64}} | {{G_F64_T}} |
| 128 | {{G_F128}} | {{G_F128_T}} |

## C.9 Raw JSON

```json
{
  "distilbert": {"acc": {{D_ACC}}, "path": "{{D_PATH}}"},
  "roberta": {"acc": {{R_ACC}}, "path": "{{R_PATH}}"},
  "gpt2": {"acc": {{G_ACC}}, "path": "cascade", "gate": [{{G_G0}}, {{G_G1}}, {{G_G2}}]}
}
```

## C.10 Reproduction Commands

```bash
python run_all_4_nasc.py --models distilbert,roberta,gpt2

for w in 16 32 64 128; do
    python run_all_4_nasc.py --window_size $w --output_suffix w$w
done
```
```

---