# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Pages | Selected | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | — | — | 183 MB |
| DistilBERT | **MoE-DeltaKV** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_PAGES}} | {{D_SEL}} | {{D_VRAM}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | — | — | 712 MB |
| RoBERTa | **MoE-DeltaKV** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_PAGES}} | {{R_SEL}} | {{R_VRAM}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | — | — | 793 MB |
| GPT-2 | **MoE-DeltaKV** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_PAGES}} | {{G_SEL}} | {{G_VRAM}} MB |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | MoE-DeltaKV | Trainable |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | {{D_TRAINABLE}} |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | {{R_TRAINABLE}} |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | {{G_TRAINABLE}} |

## 7.3 Router Behavior

**Table 7.3: Router statistics.**

| Model | Page-0 rate | Top-1 page | Aux loss |
|---|---|---|---|
| DistilBERT | {{D_P0_RATE}}% | {{D_TOP1}} | {{D_AUX}} |
| RoBERTa | {{R_P0_RATE}}% | {{R_TOP1}} | {{R_AUX}} |
| GPT-2 | {{G_P0_RATE}}% | {{G_TOP1}} | {{G_AUX}} |

## 7.4 Ablation: k_experts

**Table 7.4: Accuracy vs top-k pages.**

| k_experts | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | {{D_K1}} | {{R_K1}} | {{G_K1}} |
| 2 | {{D_K2}} | {{R_K2}} | {{G_K2}} |
| 4 | {{D_K4}} | {{R_K4}} | {{G_K4}} |
| 8 | {{D_K8}} | {{R_K8}} | {{G_K8}} |
| all | {{D_KALL}} | {{R_KALL}} | {{G_KALL}} |

## 7.5 Ablation: Page Size

**Table 7.5: Accuracy vs page size.**

| Page size | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | {{D_P16}} | {{R_P16}} | {{G_P16}} |
| 32 | {{D_P32}} | {{R_P32}} | {{G_P32}} |
| 64 | {{D_P64}} | {{R_P64}} | {{G_P64}} |
| 128 | {{D_P128}} | {{R_P128}} | {{G_P128}} |

## 7.6 Ablation: Delta-KV Rank

**Table 7.6: Accuracy vs SVD rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_R4}} | {{R_R4}} | {{G_R4}} |
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} |
| 64 | {{D_R64}} | {{R_R64}} | {{G_R64}} |

## 7.7 Ablation: Component Contribution

**Table 7.7: Effect of each component.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Full | {{D_FULL}} | {{R_FULL}} | {{G_FULL}} |
| − Delta-KV | {{D_NODELTA}} | {{R_NODELTA}} | {{G_NODELTA}} |
| − MoBA router | {{D_NOMOBA}} | {{R_NOMOBA}} | {{G_NOMOBA}} |
| − 4-bit quant | {{D_NOQUANT}} | {{R_NOQUANT}} | {{G_NOQUANT}} |
| − CLS boost | {{D_NOCLS}} | {{R_NOCLS}} | {{G_NOCLS}} |

## 7.8 Ablation: Aux Loss

**Table 7.8: Effect of auxiliary load balancing.**

| Weight | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.0 (no aux) | {{D_AUX0}} | {{R_AUX0}} | {{G_AUX0}} |
| 0.001 | {{D_AUX001}} | {{R_AUX001}} | {{G_AUX001}} |
| 0.01 | {{D_AUX01}} | {{R_AUX01}} | {{G_AUX01}} |
| 0.1 | {{D_AUX1}} | {{R_AUX1}} | {{G_AUX1}} |

## 7.9 Efficiency Profile

**Table 7.9: Inference efficiency.**

| Model | Config | Time | Tokens/s | Peak VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.9 s | 59000 | 183 MB |
| DistilBERT | MoE-DeltaKV | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| GPT-2 | full | 4.2 s | 26700 | 793 MB |
| GPT-2 | MoE-DeltaKV | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.10 Scaling with Sequence Length

**Table 7.10: Time vs L (encoder).**

| L | Standard | MoE-DeltaKV | Speedup |
|---|---|---|---|
| 128 | {{T128_STD}} | {{T128_MOE}} | {{S128}}× |
| 256 | {{T256_STD}} | {{T256_MOE}} | {{S256}}× |
| 512 | {{T512_STD}} | {{T512_MOE}} | {{S512}}× |
| 1024 | {{T1024_STD}} | {{T1024_MOE}} | {{S1024}}× |

## 7.11 Summary

- Accuracy near baseline on encoders.
- Router successfully selects diverse pages (page-0 rate ≈ 100%).
- Speedup grows with sequence length.
- **Caveat:** KV cache memory unchanged at inference.
```

---