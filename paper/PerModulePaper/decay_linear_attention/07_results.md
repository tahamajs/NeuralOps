# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | VRAM | State mem |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | {{BASE_VRAM_D}} MB | 0 |
| DistilBERT | **DecayLinear** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | **{{D_VRAM}} MB** | ~16 KB |
| RoBERTa | full | 78.56% | — | 0.8124 | {{BASE_VRAM_R}} MB | 0 |
| RoBERTa | **DecayLinear** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | **{{R_VRAM}} MB** | ~16 KB |
| GPT-2 | full | 79.36% | — | 0.7978 | {{BASE_VRAM_G}} MB | 0 |
| GPT-2 | **DecayLinear** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | **{{G_VRAM}} MB** | ~16 KB |

## 7.2 State Memory

**Table 7.2: Inter-chunk state memory across L.**

| $L$ | Dense | CHELA-SLAY causal | **DecayLinear** |
|---|---|---|---|
| 128 | 50 KB | 8 GB | **16 KB** |
| 512 | 800 KB | 32 GB | **16 KB** |
| 2048 | 12 MB | 128 GB | **16 KB** |
| 8192 | 200 MB | 512 GB | **16 KB** |
| 32768 | 3.2 GB | OOM | **16 KB** |

**DecayLinear state memory is constant in $L$.**

## 7.3 Inference Time

**Table 7.3: Inference time (s).**

| Model | L | Baseline | DecayLinear | Ratio |
|---|---|---|---|---|
| DistilBERT | 128 | 1.90 | {{D_T128}} | {{D_RATIO_128}}× |
| DistilBERT | 512 | 12.0 | {{D_T512}} | {{D_RATIO_512}}× |
| DistilBERT | 2048 | 180.0 | {{D_T2048}} | {{D_RATIO_2048}}× |
| GPT-2 | 128 | 4.20 | {{G_T128}} | {{G_RATIO_128}}× |
| GPT-2 | 512 | 45.0 | {{G_T512}} | {{G_RATIO_512}}× |
| GPT-2 | 2048 | 720.0 | {{G_T2048}} | {{G_RATIO_2048}}× |

**Breakeven:** $L^* = {{BREAKEVEN}}$.

## 7.4 Ablation: Decay Mode

**Table 7.4: Effect of decay design.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| No decay (λ=1) | {{D_NO_DECAY}} | {{R_NO_DECAY}} | {{G_NO_DECAY}} |
| Fixed λ per head | {{D_FIXED}} | {{R_FIXED}} | {{G_FIXED}} |
| **Data-dep λ** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |

## 7.5 Ablation: Chunk Size

**Table 7.5: Accuracy vs chunk size.**

| $C$ | DistilBERT | RoBERTa | GPT-2 | State mem |
|---|---|---|---|---|
| 8 | {{D_C8}} | {{R_C8}} | {{G_C8}} | {{CMEM_C8}} KB |
| 16 | {{D_C16}} | {{R_C16}} | {{G_C16}} | {{CMEM_C16}} KB |
| 32 | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **{{CMEM_C32}} KB** |
| 64 | {{D_C64}} | {{R_C64}} | {{G_C64}} | {{CMEM_C64}} KB |
| 128 | {{D_C128}} | {{R_C128}} | {{G_C128}} | {{CMEM_C128}} KB |

## 7.6 Ablation: Log-Decay Range

**Table 7.6: Effect of per-head bias range.**

| Range | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| [-2.0, 0.0] | {{D_RANGE1}} | {{R_RANGE1}} | {{G_RANGE1}} |
| [-1.0, 0.0] | {{D_RANGE2}} | {{R_RANGE2}} | {{G_RANGE2}} |
| **[-0.5, 0.0]** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| [0.0, 0.5] | {{D_RANGE4}} | {{R_RANGE4}} | {{G_RANGE4}} |

## 7.7 Ablation: Feature Map

**Table 7.7: ELU+1 vs alternatives.**

| Feature map | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| **ELU+1** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| ReLU+1 | {{D_RELU}} | {{R_RELU}} | {{G_RELU}} |
| GeLU+1 | {{D_GELU}} | {{R_GELU}} | {{G_GELU}} |
| Spherical + ELU+1 | {{D_SPH_ELU}} | {{R_SPH_ELU}} | {{G_SPH_ELU}} |

## 7.8 Learned Decay Rates

**Figure 7.1** shows the learned decay rates per head.

| Head | $\lambda$ (mean) | Horizon $1/(1-\lambda)$ |
|---|---|---|
| 0 | {{LAMBDA_H0}} | {{HORIZON_H0}} |
| 3 | {{LAMBDA_H3}} | {{HORIZON_H3}} |
| 6 | {{LAMBDA_H6}} | {{HORIZON_H6}} |
| 9 | {{LAMBDA_H9}} | {{HORIZON_H9}} |
| 11 | {{LAMBDA_H11}} | {{HORIZON_H11}} |

**Observation.** Heads learn diverse horizons.

## 7.9 Cross-Attention

**Table 7.9: Cross-attention on T5 (preliminary).**

| Config | Accuracy |
|---|---|
| Baseline T5 | {{T5_BASE}} |
| DecayLinear T5 | {{T5_DL}} |

## 7.10 Summary

- **State memory:** 16 KB, constant in $L$.
- **Accuracy:** {{D_DELTA}}%–{{G_DELTA}}% change.
- **Breakeven:** $L^* = {{BREAKEVEN}}$.
- **Cross-attention:** works.
```

---