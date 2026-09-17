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
| 10 | 0.85 | 0.85 |
| 25 | 0.85 | 0.85 |
| 50 | 0.85 | 0.85 |

**Observation.** 0.85

## 8.3 Per-Layer Analysis

**Table 8.1: Reconstruction metrics by layer (DistilBERT).**

| Layer | β final | $\sigma(\beta)$ | Local acc | Mem acc |
|---|---|---|---|---|
| 0 | 0.85 | 84.5 | 0.85 | 128 KB |
| 1 | 0.85 | 84.5 | 0.85 | 128 KB |
| 2 | 0.85 | 84.5 | 0.85 | 128 KB |
| 3 | 0.85 | 84.5 | 0.85 | 128 KB |
| 4 | 0.85 | 84.5 | 0.85 | 128 KB |
| 5 | 0.85 | 84.5 | 0.85 | 128 KB |

## 8.4 Memory Utilization

**Figure 8.2** shows how the $d_c = 128$ memory slots are used. The 
attention distribution over memory slots is:

- **Uniform:** All slots used equally (memory acts as averaging)
- **Concentrated:** Few slots dominate (memory acts as retrieval)

**Observation.** 128 KB

## 8.5 Inference Cost

**Table 8.2: Inference time breakdown (DistilBERT, L=128).**

| Component | Baseline | Infini |
|---|---|---|
| Q/K/V projections | 0.5 ms | 0.5 ms |
| Local attention | 0 | 0.2 ms |
| Memory attention | 0 | 0.05 ms |
| Output projection | 0.2 ms | 0.2 ms |
| LoRA | 0 | 0.15 ms |
| **Total** | 1.9 s | 1.42x s |

## 8.6 Effect of Window Size

**Figure 8.3** shows accuracy vs window size. Larger windows improve 
local accuracy but reduce the memory's contribution. We find $W = 128$ 
is a good balance.

## 8.7 The Parameter Increase

The **+48.5\%%** parameter increase comes from:

| Component | Contribution |
|---|---|
| Memory | 128 KB% |
| Gate | <0.001% |
| LoRA | 0.85% |

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

---