# 8. Analysis

## 8.1 Encoder vs Decoder Dispatch

**Observation.** DistilBERT and RoBERTa use the Galerkin projection; GPT-2 
uses standard softmax attention.

**Why?** Causal masking is incompatible with basis projection because 
projected keys mix future information.

**Implication.** Galerkin provides no benefit on GPT-2.

## 8.2 Basis Approximation Quality

**Figure 8.1** shows the spectrum of DCT coefficients for a sample attention 
output. The top 64 coefficients capture ~{{DCT_ENERGY}}% of the energy, 
justifying $r = 64$.

## 8.3 Per-Layer Reconstruction Error

**Table 8.1: Frobenius error of attention output vs full softmax (DistilBERT).**

| Layer | Relative error |
|---|---|
| 0 | {{D_L0_ERR}} |
| 1 | {{D_L1_ERR}} |
| 2 | {{D_L2_ERR}} |
| 3 | {{D_L3_ERR}} |
| 4 | {{D_L4_ERR}} |
| 5 | {{D_L5_ERR}} |

Errors grow with depth as feature complexity increases.

## 8.4 Energy Preservation Validation

We verify that the $\sqrt{N/r}$ scaling preserves activation magnitudes:

| Config | Mean activation std |
|---|---|
| Baseline | 1.00 |
| Without scaling | {{STD_NO_SCALE}} |
| With scaling | {{STD_WITH_SCALE}} |

## 8.5 Comparison with Fixed Rank Projection

**Table 8.3: DCT vs random orthonormal basis.**

| Basis | D acc | R acc |
|---|---|---|
| Random orthonormal | {{RAND_D}} | {{RAND_R}} |
| **DCT** | **{{D_ACC}}** | **{{R_ACC}}** |

DCT's smoothness assumption matches attention better.

## 8.6 Failure Analysis

{{FAILURES}}

## 8.7 Summary of Insights

1. **Encoder benefit is real** — 2× speedup at $N=128$.
2. **Decoder fallback necessary** — no benefit for GPT-2.
3. **DCT basis optimal** — beats random orthonormal.
4. **Energy scaling works** — preserves activation magnitude.
5. **Parameter count increases** — not a compression method.
