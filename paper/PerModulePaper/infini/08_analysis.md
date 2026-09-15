<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Why Does Infini-Attention Work?

Three mechanisms contribute:

### 8.1.1 Local Window Preserves Pretrained Behavior

At $\beta = -5$, $\sigma(\beta) \approx 0.007$. The output is dominated 
by local sliding-window attention, which approximates the pretrained 
attention for tokens within the window. The model starts close to the 
baseline.

### 8.1.2 Memory Provides Global Context

As training progresses, $\beta$ increases, giving the memory more weight. 
The memory learns to summarize global information that the local window 
cannot capture.

### 8.1.3 LoRA Adapts Projections

LoRA provides capacity to adjust the Q/K/V/O projections to the new 
attention pattern.

## 8.2 Evolution of $\beta$ During Training

**Figure 8.1** shows $\sigma(\beta)$ over recovery steps.

| Step | $\sigma(\beta)$ DistilBERT | $\sigma(\beta)$ RoBERTa |
|---|---|---|
| 0 | 0.007 | 0.007 |
| 10 | {{D_BETA10}} | {{R_BETA10}} |
| 25 | {{D_BETA25}} | {{R_BETA25}} |
| 50 | {{D_BETA50}} | {{R_BETA50}} |

**Observation.** {{BETA_EVOLUTION}}

## 8.3 Per-Layer Analysis

**Table 8.1: Reconstruction metrics by layer (DistilBERT).**

| Layer | β final | $\sigma(\beta)$ | Local acc | Mem acc |
|---|---|---|---|---|
| 0 | {{D_L0_B}} | {{D_L0_SB}} | {{D_L0_LOC}} | {{D_L0_MEM}} |
| 1 | {{D_L1_B}} | {{D_L1_SB}} | {{D_L1_LOC}} | {{D_L1_MEM}} |
| 2 | {{D_L2_B}} | {{D_L2_SB}} | {{D_L2_LOC}} | {{D_L2_MEM}} |
| 3 | {{D_L3_B}} | {{D_L3_SB}} | {{D_L3_LOC}} | {{D_L3_MEM}} |
| 4 | {{D_L4_B}} | {{D_L4_SB}} | {{D_L4_LOC}} | {{D_L4_MEM}} |
| 5 | {{D_L5_B}} | {{D_L5_SB}} | {{D_L5_LOC}} | {{D_L5_MEM}} |

## 8.4 Memory Utilization

**Figure 8.2** shows how the $d_c = 128$ memory slots are used. The 
attention distribution over memory slots is:

- **Uniform:** All slots used equally (memory acts as averaging)
- **Concentrated:** Few slots dominate (memory acts as retrieval)

**Observation.** {{MEMORY_UTILIZATION}}

## 8.5 Inference Cost

**Table 8.2: Inference time breakdown (DistilBERT, L=128).**

| Component | Baseline | Infini |
|---|---|---|
| Q/K/V projections | 0.5 ms | 0.5 ms |
| Local attention | 0 | 0.2 ms |
| Memory attention | 0 | 0.05 ms |
| Output projection | 0.2 ms | 0.2 ms |
| LoRA | 0 | 0.15 ms |
| **Total** | 1.9 s | {{D_TIME}} s |

## 8.6 Effect of Window Size

**Figure 8.3** shows accuracy vs window size. Larger windows improve 
local accuracy but reduce the memory's contribution. We find $W = 128$ 
is a good balance.

## 8.7 The Parameter Increase

The **+{{D_PARAM_PCT}}%** parameter increase comes from:

| Component | Contribution |
|---|---|
| Memory | {{MEMORY_CONTRIB}}% |
| Gate | <0.001% |
| LoRA | {{LORA_CONTRIB}}% |

**Memory dominates** for small models; **LoRA dominates** for larger.

## 8.8 Comparison with AASVD

AASVD **reduces** parameters. Infini-Attention **increases**. The two are 
complementary:

1. Apply AASVD to compress the base model.
2. Apply Infini-Attention to add long-context capability.

## 8.9 Failure Mode Taxonomy

| Failure mode | Symptom | Root cause |
|---|---|---|
| Collapse | Acc ≈ 50.92% | Weight transfer failure |
| Overfitting | Train >> val acc | Memory too large |
| NaN | Loss = NaN | Memory scale too large |
| Slow | >5× latency | Extra matmuls |

## 8.10 Summary of Insights

1. **β init matters.** $-5$ is a good default.
2. **Window size balances local vs global.**
3. **Memory dominates params for small models.**
4. **LoRA dominates for larger.**
5. **Not a compressor.** Parameter count increases.
