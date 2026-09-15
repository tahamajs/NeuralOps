<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Config | GQA with G={{D_KV}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Params before | 66,955,010 |
| Params after | {{D_PARAMS}} |
| Reduction | {{D_P_RATIO}}× |
| KV/token | {{D_KV_PER_TOKEN}} KB |
| KV ratio | {{D_KV_R}}× |
| Peak VRAM | {{D_VRAM}} MB |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Config | GQA with G={{R_KV}} |
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Params before | 124,647,170 |
| Params after | {{R_PARAMS}} |
| Reduction | {{R_P_RATIO}}× |
| KV/token | {{R_KV_PER_TOKEN}} KB |
| KV ratio | {{R_KV_R}}× |
| Peak VRAM | {{R_VRAM}} MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Config | GQA with G={{G_KV}} |
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Params before | 124,441,344 |
| Params after | {{G_PARAMS}} |
| Reduction | {{G_P_RATIO}}× |
| KV/token | {{G_KV_PER_TOKEN}} KB |
| KV ratio | {{G_KV_R}}× |
| Peak VRAM | {{G_VRAM}} MB |

## C.4 KV Heads Ablation

### DistilBERT
| $G$ | Acc | KV/token (KB) | Ratio |
|---|---|---|---|
| 12 | 90.71% | 36.9 | 1.00× |
| 6 | {{D_G6}} | 18.4 | 2.00× |
| 4 | {{D_G4}} | 12.3 | 3.00× |
| 3 | {{D_G3}} | 9.2 | 4.00× |
| 2 | {{D_G2}} | 6.1 | 6.00× |
| 1 | {{D_G1}} | 3.1 | 12.0× |

(Similar tables for RoBERTa and GPT-2)

## C.5 LoRA Rank Ablation

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_LR4}} | {{R_LR4}} | {{G_LR4}} |
| 8 | {{D_LR8}} | {{R_LR8}} | {{G_LR8}} |
| 16 | {{D_LR16}} | {{R_LR16}} | {{G_LR16}} |
| 32 | {{D_LR32}} | {{R_LR32}} | {{G_LR32}} |

## C.6 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## C.7 Training Loss Curves

### DistilBERT
```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
10    {{D_S10}}
...
50    {{D_S50}}
```

(Similar for RoBERTa, GPT-2)

## C.8 Raw JSON

```json
{
  "gqa": {
    "distilbert": {
      "num_kv_heads": {{D_KV}},
      "accuracy_pct": {{D_ACC}},
      "params_before": 66955010,
      "params_after": {{D_PARAMS}},
      "kv_bytes_per_token": {{D_KV_BYTES}},
      "kv_ratio": {{D_KV_R}}
    },
    "roberta": {...},
    "gpt2": {...}
  }
}
```

## C.9 Reproduction Commands

```bash
# Main
python run_all_4_gqa.py --num_kv_heads 6

# Rank ablation
for kv in 1 2 3 4 6 12; do
    python run_all_4_gqa.py --num_kv_heads $kv
done
```
