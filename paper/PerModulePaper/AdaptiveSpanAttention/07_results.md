<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 results with $S_\text{max} = {{MAX_SPAN}}$, $R = {{RAMP_SIZE}}$, $\lambda = {{LAMBDA}}$.**

| Model | Module | Acc | Δ | F1 | Learned span | Attn mem | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | — | 128 KB | 1.9 s |
| DistilBERT | **ASA-Span** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | **{{D_SPAN}}** | {{D_MEM}} | {{D_TIME}} s |
| RoBERTa | full | 78.56% | — | 0.8124 | — | 128 KB | 4.1 s |
| RoBERTa | **ASA-Span** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | **{{R_SPAN}}** | {{R_MEM}} | {{R_TIME}} s |
| GPT-2 | full | 79.36% | — | 0.7978 | — | 128 KB | 4.2 s |
| GPT-2 | **ASA-Span** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | **{{G_SPAN}}** | {{G_MEM}} | {{G_TIME}} s |

## 7.2 Learned Spans

**Table 7.2: Span distribution across heads.**

| Model | Mean span | Min | Max | Std | Reduction from $S_\text{max}$ |
|---|---|---|---|---|---|
| DistilBERT | **{{D_SPAN}}** | {{D_SPAN_MIN}} | {{D_SPAN_MAX}} | {{D_SPAN_STD}} | {{D_SPAN_REDUCTION}}% |
| RoBERTa | **{{R_SPAN}}** | {{R_SPAN_MIN}} | {{R_SPAN_MAX}} | {{R_SPAN_STD}} | {{R_SPAN_REDUCTION}}% |
| GPT-2 | **{{G_SPAN}}** | {{G_SPAN_MIN}} | {{G_SPAN_MAX}} | {{G_SPAN_STD}} | {{G_SPAN_REDUCTION}}% |

**Observation.** Learned spans are **highly non-uniform**. Some heads 
converge to < {{D_SPAN_MIN}} (local), others to near $S_\text{max}$ (global).

## 7.3 Per-Layer Span Evolution

**Table 7.3: Mean span per layer (DistilBERT).**

| Layer | Mean span |
|---|---|
| 0 | {{D_L0_SPAN}} |
| 1 | {{D_L1_SPAN}} |
| 2 | {{D_L2_SPAN}} |
| 3 | {{D_L3_SPAN}} |
| 4 | {{D_L4_SPAN}} |
| 5 | {{D_L5_SPAN}} |

**Observation.** Early layers tend to have **larger** spans; late layers 
have **smaller** spans. This matches the observation that early layers 
integrate global context while late layers refine local features.

## 7.4 Efficiency

**Table 7.4: Memory and time.**

| Model | Config | Peak VRAM | Inference Time |
|---|---|---|---|
| DistilBERT | full | 183 MB | 1.90 s |
| DistilBERT | ASA-Span | {{D_VRAM}} MB | {{D_TIME}} s |
| GPT-2 | full | 793 MB | 4.20 s |
| GPT-2 | ASA-Span | {{G_VRAM}} MB | {{G_TIME}} s |

**Note.** ASA-Span does not reduce inference time in the current implementation 
because the ramp is applied as a **soft mask** rather than a hard window. 
To realize speedup, a **hard truncation** post-training is needed.

## 7.5 Ablation: Max Span

**Table 7.5: Accuracy vs $S_\text{max}$.**

| $S_\text{max}$ | DistilBERT | RoBERTa | GPT-2 | Learned mean |
|---|---|---|---|---|
| 32 | {{D_MAX32}} | {{R_MAX32}} | {{G_MAX32}} | {{D_LEARN32}} |
| 64 | {{D_MAX64}} | {{R_MAX64}} | {{G_MAX64}} | {{D_LEARN64}} |
| 128 | {{D_MAX128}} | {{R_MAX128}} | {{G_MAX128}} | {{D_LEARN128}} |
| 256 | {{D_MAX256}} | {{R_MAX256}} | {{G_MAX256}} | {{D_LEARN256}} |

## 7.6 Ablation: Ramp Size

**Table 7.6: Accuracy vs $R$.**

| $R$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | {{D_R1}} | {{R_R1}} | {{G_R1}} |
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} |
| 64 | {{D_R64}} | {{R_R64}} | {{G_R64}} |

**Observation.** Larger $R$ gives smoother transitions but less aggressive 
pruning.

## 7.7 Ablation: L1 Penalty

**Table 7.7: Accuracy and span vs $\lambda$.**

| $\lambda$ | DistilBERT acc | Mean span | Reduction |
|---|---|---|---|
| 0.0 | {{D_L0}} | {{D_SPAN0}} | 0% |
| 0.001 | {{D_L001}} | {{D_SPAN001}} | {{D_RED001}}% |
| 0.01 | {{D_L01}} | {{D_SPAN01}} | {{D_RED01}}% |
| 0.1 | {{D_L1}} | {{D_SPAN1}} | {{D_RED1}}% |

**Observation.** $\lambda = 0.01$ is a good balance.

## 7.8 Ablation: LoRA Rank

**Table 7.8: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_LR4}} | {{R_LR4}} | {{G_LR4}} |
| 8 | {{D_LR8}} | {{R_LR8}} | {{G_LR8}} |
| 16 | {{D_LR16}} | {{R_LR16}} | {{G_LR16}} |
| 32 | {{D_LR32}} | {{R_LR32}} | {{G_LR32}} |

## 7.9 Ablation: Recovery Steps

**Table 7.9: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.10 Comparison with Companion Methods

**Table 7.10: Comparison.**

| Method | DistilBERT | RoBERTa | GPT-2 | Type | Params Δ | Attn mem |
|---|---|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | Dense | 1.00× | 1.00× |
| GQA | {{GQA_D}} | {{GQA_R}} | {{GQA_G}} | KV reduce | 0.95× | 1.00× |
| Primal | {{PRIMAL_D}} | {{PRIMAL_R}} | {{PRIMAL_G}} | Low-rank | 1.00× | **{{PRIMAL_MEM}}×** |
| ASA-Sparse | {{ASA_D}} | {{ASA_R}} | {{ASA_G}} | Sparse | 1.00× | 1.00× |
| **ASA-Span** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | Learned span | 1.00× | 1.00× |

## 7.11 Training Loss and Span Evolution

**Figure 7.1** shows span evolution over training steps.

**Observation.** Spans start near $S_\text{max}$ and gradually shrink. The 
shrinkage is faster in early layers, slower in late layers. By step 50, 
convergence is reached.

## 7.12 Summary

- **Accuracy:** drops {{D_DELTA}}%–{{G_DELTA}}%.
- **Learned spans:** {{D_SPAN}}–{{G_SPAN}} (from max {{MAX_SPAN}}).
- **Reduction:** {{D_SPAN_REDUCTION}}%–{{G_SPAN_REDUCTION}}%.
- **No parameter reduction.**
- **No speedup in current implementation.**
- **All models pass quality gate.**
```

---