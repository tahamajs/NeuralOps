<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS}} |
| Param Δ | +{{D_PARAM_DELTA}} |
| Param Δ % | +{{D_PARAM_PCT}}% |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Sketch rank | {{RANK}} |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS}} |
| Param Δ | +{{R_PARAM_DELTA}} |
| Param Δ % | +{{R_PARAM_PCT}}% |
| Accuracy | {{R_ACC}}% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS}} |
| Param Δ | +{{G_PARAM_DELTA}} |
| Param Δ % | +{{G_PARAM_PCT}}% |
| Accuracy | {{G_ACC}}% |

## C.4 Sketch Rank Ablation

| $r$ | DistilBERT | RoBERTa | GPT-2 | Theoretical speedup |
|---|---|---|---|---|
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} | 8× |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} | 4× |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} | 2× |
| 64 | {{D_R64}} | {{R_R64}} | {{G_R64}} | 1× |

## C.5 LoRA Rank Ablation

| $r_L$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | {{D_LORA0}} | {{R_LORA0}} | {{G_LORA0}} |
| 8 | {{D_LORA8}} | {{R_LORA8}} | {{G_LORA8}} |
| 16 | {{D_LORA16}} | {{R_LORA16}} | {{G_LORA16}} |
| 32 | {{D_LORA32}} | {{R_LORA32}} | {{G_LORA32}} |

## C.6 Sketch Quality

| $r$ | Score correlation | MSE |
|---|---|---|
| 8 | {{CORR8}} | {{MSE8}} |
| 16 | {{CORR16}} | {{MSE16}} |
| 32 | {{CORR32}} | {{MSE32}} |
| 64 | 1.00 | 0 |

## C.7 Efficiency at Different Sequence Lengths

| $L$ | Baseline time | RPC time (r=32) | Speedup |
|---|---|---|---|
| 64 | {{B64}} | {{RPC64}} | {{S64}}× |
| 128 | 1.90 s | {{RPC128}} | {{S128}}× |
| 256 | {{B256}} | {{RPC256}} | {{S256}}× |
| 512 | {{B512}} | {{RPC512}} | {{S512}}× |
| 1024 | {{B1024}} | {{RPC1024}} | {{S1024}}× |

## C.8 Training Loss Curves

### DistilBERT
```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
...
50    {{D_S50}}
```

## C.9 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": {{D_PARAMS}},
    "param_delta": {{D_PARAM_DELTA}},
    "accuracy_pct": {{D_ACC}},
    "sketch_rank": {{RANK}},
    "score_correlation": {{CORR32}}
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.10 Reproduction Commands

```bash
# Full benchmark
python run_all_4_rpc.py --models distilbert,roberta,gpt2

# Rank sweep
for r in 8 16 32 64; do
    python run_all_4_rpc.py --rank $r --output_suffix r$r
done
```

---
