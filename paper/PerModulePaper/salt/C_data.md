<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Params before | 66,955,010 |
| Params after | {{D_PARAMS}} |
| Reduction | {{D_PARAM_RED}}× |
| Storage (MB) | {{D_STORAGE}} |
| Peak VRAM (MB) | {{D_VRAM}} |
| Inference time (s) | {{D_TIME}} |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Params before | 124,647,170 |
| Params after | {{R_PARAMS}} |
| Reduction | {{R_PARAM_RED}}× |
| Storage (MB) | {{R_STORAGE}} |
| Peak VRAM (MB) | {{R_VRAM}} |
| Inference time (s) | {{R_TIME}} |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Params before | 124,441,344 |
| Params after | {{G_PARAMS}} |
| Reduction | {{G_PARAM_RED}}× |
| Storage (MB) | {{G_STORAGE}} |
| Peak VRAM (MB) | {{G_VRAM}} |
| Inference time (s) | {{G_TIME}} |

## C.4 Rank Ablation (DistilBERT)

| Rank | Acc | Params/layer | Reduction |
|---|---|---|---|
| 4 | {{D_R4}} | 12,336 | 47.8× |
| 8 | {{D_R8}} | 24,656 | 23.9× |
| 16 | {{D_R16}} | 49,296 | 12.0× |
| 32 | {{D_R32}} | 98,576 | 5.98× |
| 64 | {{D_R64}} | 197,136 | 2.99× |

## C.5 Gamma Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| γ trainable | {{D_G}} | {{R_G}} | {{G_G}} |
| γ frozen | {{D_NOG}} | {{R_NOG}} | {{G_NOG}} |

## C.6 AB Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| AB trainable | {{D_AB}} | {{R_AB}} | {{G_AB}} |
| AB frozen | {{D_NOAB}} | {{R_NOAB}} | {{G_NOAB}} |

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

## C.8 Singular Value Statistics

### DistilBERT, layer 3

| k | σ_k | Energy fraction |
|---|---|---|
| 1 | {{D_SIGMA_1}} | {{D_E_1}}% |
| 2 | {{D_SIGMA_2}} | {{D_E_2}}% |
| ... | ... | ... |
| 16 | {{D_SIGMA_16}} | {{D_E_16}}% |
| 17+ | — | {{D_E_REST}}% |

## C.9 Raw JSON

```json
{
  "distilbert": {
    "accuracy": {{D_ACC}},
    "params_before": 66955010,
    "params_after": {{D_PARAMS}},
    "reduction": {{D_PARAM_RED}},
    "storage_mb": {{D_STORAGE}},
    "peak_vram_mb": {{D_VRAM}}
  },
  ...
}
```

## C.10 Reproduction

```bash
python run_all_4_salt.py --models distilbert,roberta,gpt2 --rank {{RANK}}
```

---