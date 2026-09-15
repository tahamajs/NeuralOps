<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Why Does MinimalPRISM Work?

Three mechanisms contribute:

### 8.1.1 Spectral Decay

Transformer weight matrices have **rapidly decaying spectra**. Figure 8.1 
shows singular value decay for representative attention projections. The 
top 32 singular values capture over **{{TOP32_ENERGY}}%** of the energy.

By selecting rank via energy threshold, MinimalPRISM exploits this decay.

### 8.1.2 Energy-Adaptive Rank

Figure 8.2 shows the effective rank distribution across layers. Early layers 
have lower rank (typically {{EARLY_RANK}}), middle layers require more 
({{MID_RANK}}), late layers again compress well ({{LATE_RANK}}).

This heterogeneity justifies energy-adaptive rank over fixed rank.

### 8.1.3 LoRA Recovery

After factorization, LoRA adapters recover **{{LORA_RECOVERY}}%** of the 
accuracy lost to truncation. Without LoRA, accuracy drops to 
**{{NO_LORA_ACC}}%**.

## 8.2 Per-Layer Reconstruction Error

**Table 8.1: Layer-wise Frobenius error (DistilBERT).**

| Layer | Q | K | V | O | Mean |
|---|---|---|---|---|---|
| 0 | {{D_L0_EQ}} | {{D_L0_EK}} | {{D_L0_EV}} | {{D_L0_EO}} | {{D_L0_EM}} |
| 1 | {{D_L1_EQ}} | {{D_L1_EK}} | {{D_L1_EV}} | {{D_L1_EO}} | {{D_L1_EM}} |
| 2 | {{D_L2_EQ}} | {{D_L2_EK}} | {{D_L2_EV}} | {{D_L2_EO}} | {{D_L2_EM}} |
| 3 | {{D_L3_EQ}} | {{D_L3_EK}} | {{D_L3_EV}} | {{D_L3_EO}} | {{D_L3_EM}} |
| 4 | {{D_L4_EQ}} | {{D_L4_EK}} | {{D_L4_EV}} | {{D_L4_EO}} | {{D_L4_EM}} |
| 5 | {{D_L5_EQ}} | {{D_L5_EK}} | {{D_L5_EV}} | {{D_L5_EO}} | {{D_L5_EM}} |

## 8.3 Contribution of Each Component

**Table 8.2: Ablation on DistilBERT.**

| Config | Accuracy | Δ |
|---|---|---|
| Baseline | 90.71% | — |
| SVD only, no LoRA | {{SVD_NOLORA}} | {{SVD_NOLORA_D}} |
| + LoRA (rank 4) | {{LORA4}} | {{LORA4_D}} |
| + LoRA (rank 8) | {{LORA8}} | {{LORA8_D}} |
| + LoRA (rank 16) | **{{D_ACC}}** | **{{D_DELTA}}** |

## 8.4 Rank Distribution Analysis

Figure 8.3 shows effective rank per layer for all three models:

- **DistilBERT:** {{D_RANK_OBS}}
- **RoBERTa:** {{R_RANK_OBS}}
- **GPT-2:** {{G_RANK_OBS}}

## 8.5 Effect of Energy Threshold

Figure 8.4 shows accuracy vs $\eta$ for all models.

| $\eta$ | DistilBERT | Avg Rank | Theoretical Bound |
|---|---|---|---|
| 0.90 | {{D_E90}} | {{D_AR90}} | 0.316 |
| 0.95 | {{D_E95}} | {{D_AR95}} | 0.224 |
| 0.99 | {{D_E99}} | {{D_AR99}} | 0.100 |
| 0.999 | {{D_E999}} | {{D_AR999}} | 0.032 |

**Observation.** Higher $\eta$ improves accuracy at the cost of compression.

## 8.6 Comparison with Fixed Rank

**Table 8.3: Energy-adaptive vs fixed rank.**

| Method | DistilBERT | Avg Rank | Ratio |
|---|---|---|---|
| Fixed rank 32 | {{FIXED32}} | 32 | {{FIXED32_R}}× |
| Fixed rank 64 | {{FIXED64}} | 64 | {{FIXED64_R}}× |
| **Energy 0.95** | **{{D_ACC}}** | {{D_AR95}} | **{{D_RATIO}}×** |

**Observation.** Energy-adaptive achieves {{ENERGY_ADVANTAGE}}% better 
accuracy at similar compression.

## 8.7 Injection Safety Analysis

### 8.7.1 Pre-Test Success Rate

The pre-test on the first layer catches:

| Failure mode | Detected? | Action |
|---|---|---|
| Missing Q weights | Yes | Abort |
| Near-zero Frobenius | Yes | Abort |
| Shape mismatch | No (handled) | Warn |
| Wrong Conv1D orientation | Partial | Warn |

### 8.7.2 Nested Encoder Correctness

**Table 8.4: Nested encoder detection accuracy.**

| Model | is_nested | Correct? |
|---|---|---|
| DistilBERT | False | ✅ |
| RoBERTa | True | ✅ |
| GPT-2 | False | ✅ |
| Qwen3.5 | False | ✅ |

## 8.8 Inference Cost Breakdown

MinimalPRISM's inference is **faster** than dense attention:

**Table 8.5: FLOPs per projection.**

| Config | FLOPs | Relative |
|---|---|---|
| Dense $768 \times 768$ | 1.18M | 1.00× |
| Factorized $r=64$ | 196K | **0.167×** |
| Factorized $r=128$ | 393K | 0.333× |

## 8.9 Comparison with Companion Methods

**Table 8.6: MinimalPRISM vs A3, AASVD, NOVA.**

| Aspect | A3 | AASVD | MinimalNOVA | MinimalPRISM |
|---|---|---|---|---|
| Weighting | Fisher | Covariance | None | None |
| Calibration | Required | Required | None | None |
| Rank | Adaptive retry | Fixed | Energy | Energy |
| Nested encoder | Buggy | Buggy | Correct | **Correct + injection-time** |
| Pre-test | No | No | Yes | Yes |

## 8.10 Failure Modes

### 8.10.1 Layers That Failed

| Model | Layer | Error | Action |
|---|---|---|---|
| {{FAIL_MODEL1}} | {{FAIL_LAYER1}} | {{FAIL_ERR1}} | Fallback |

## 8.11 Summary of Insights

1. **Energy-adaptive rank matches fixed rank accuracy** at similar compression.
2. **Spectral decay is strong** in attention projections.
3. **LoRA recovery is essential.**
4. **Effective rank varies** across layers and models.
5. **No calibration needed.**
6. **Injection safety matters** — pre-test prevents catastrophic failures.

---
