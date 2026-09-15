<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results (N = 128)

**Table 7.1: Accuracy on SST-2 validation.**

| Model | Module | Acc | Δ | F1 | Path | Time | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | — | 1.90 s | 183 MB |
| DistilBERT | **Nyström** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | Dense | {{D_TIME}} s | {{D_VRAM}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | — | 4.10 s | 712 MB |
| RoBERTa | **Nyström** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | Dense | {{R_TIME}} s | {{R_VRAM}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | — | 4.20 s | 793 MB |
| GPT-2 | **Nyström** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | Dense causal | {{G_TIME}} s | {{G_VRAM}} MB |

**Key observation.** At $N = 128$, all models use the dense fallback. Accuracy 
and efficiency match the baseline. **The Nyström branch is not activated.**

## 7.2 Long Sequence Results (N = 512)

**Table 7.2: Accuracy and efficiency at N = 512.**

| Model | Config | Acc | Time | VRAM | Path |
|---|---|---|---|---|---|
| DistilBERT | full | {{D_L512_FULL_ACC}} | {{D_L512_FULL_TIME}} | {{D_L512_FULL_VRAM}} | Dense |
| DistilBERT | Nyström ($m=32$) | {{D_L512_NYS_ACC}} | {{D_L512_NYS_TIME}} | {{D_L512_NYS_VRAM}} | **Nyström** |
| DistilBERT | Nyström ($m=64$) | {{D_L512_NYS64_ACC}} | {{D_L512_NYS64_TIME}} | {{D_L512_NYS64_VRAM}} | Nyström |

**Observation.** {{LONG_OBS}}

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency (N=128, dense).**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | Nyström | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | Nyström | {{R_TIME}} s | {{R_TPS}} | {{R_VRAM}} MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | Nyström | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.4 Ablation: Number of Landmarks

**Table 7.4: Accuracy vs num_landmarks (N=512, DistilBERT).**

| $m$ | Acc | Time | VRAM | Speedup |
|---|---|---|---|---|
| 16 | {{D_M16}} | {{D_M16_T}} | {{D_M16_V}} | {{D_M16_S}}× |
| 32 | {{D_M32}} | {{D_M32_T}} | {{D_M32_V}} | {{D_M32_S}}× |
| 64 | {{D_M64}} | {{D_M64_T}} | {{D_M64_V}} | {{D_M64_S}}× |
| 128 | {{D_M128}} | {{D_M128_T}} | {{D_M128_V}} | {{D_M128_S}}× |

## 7.5 Ablation: Pinv Iterations

**Table 7.5: Accuracy vs pinv_iterations (N=512, DistilBERT, m=32).**

| Iters | Acc | Time |
|---|---|---|
| 1 | {{D_I1}} | {{D_I1_T}} |
| 2 | {{D_I2}} | {{D_I2_T}} |
| 3 | {{D_I3}} | {{D_I3_T}} |
| 5 | {{D_I5}} | {{D_I5_T}} |

## 7.6 Ablation: Short Sequence Threshold

**Table 7.6: Path activation threshold.**

| Threshold | Path at N=128 | Path at N=256 |
|---|---|---|
| 64 | Nyström | Nyström |
| 128 | Dense | Nyström |
| 256 | Dense | Dense |

## 7.7 Comparison with Companion Methods

**Table 7.7: Nyström vs other methods (N=128).**

| Method | DistilBERT | Reduces |
|---|---|---|
| full | 90.71% | — |
| MinimalSVD | {{MINSVD_D}} | Parameters |
| SVD-LLM V2 | {{SVDLLM_D}} | Parameters |
| Dobi-SVD | {{DOBI_D}} | Parameters |
| MLA | {{MLA_D}} | KV cache |
| **Nyström** | **{{D_ACC}}** | **Attention compute** |

**Note.** At $N=128$, Nyström's benefits are not visible. All methods 
appear similar in accuracy.

## 7.8 Training Curves

**Figure 7.1** shows LoRA recovery loss curves.

- DistilBERT: `{{D_START_LOSS}}` → `{{D_END_LOSS}}`
- RoBERTa: `{{R_START_LOSS}}` → `{{R_END_LOSS}}`
- GPT-2: `{{G_START_LOSS}}` → `{{G_END_LOSS}}`

## 7.9 Summary

- **At N=128:** Nyström uses dense fallback; matches baseline.
- **At N=512:** Nyström branch activates; achieves **{{LONG_SPEEDUP}}×** 
  speedup at {{LONG_ACC_DROP}}% accuracy cost.
- **Portability:** Nyströmformer ports to all four architectures.
- **No parameter or KV-cache reduction** — compute only.

---
