<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Why Does the Encoder Path Fail?

Three hypotheses, tested empirically.

### 8.1.1 H1: Truncation Loses Energy

**Claim.** The truncated DCT basis captures insufficient energy.

**Test.** Measure energy captured:

$$
E(r) = \frac{\|\Phi_r^\top K\|_F^2}{\|K\|_F^2}
\tag{8.1}
$$

for $r \in \{8, 16, 32, 64, 128\}$ on real DistilBERT activations.

**Result.** At $r = 64$, $E \approx {{D_E_R64}}$. At $r = 128$, $E \approx {{D_E_R128}}$.

**Verdict.** {{H1_VERDICT}}

### 8.1.2 H2: Fixed Basis Cannot Adapt

**Claim.** Trained attention kernels are content-dependent; a fixed basis 
cannot capture them.

**Test.** Measure attention entropy vs DCT energy distribution:

- Attention entropy (baseline): {{BASE_ENTROPY}}
- DCT low-frequency energy fraction: {{DCT_LOW_FREQ}}

**Result.** {{H2_RESULT}}

**Verdict.** This is the **primary failure mode**.

### 8.1.3 H3: Energy Correction Overshoots

**Claim.** The $\sqrt{N/r}$ correction over-amplifies.

**Test.** Compare with and without correction:

| Correction | DistilBERT |
|---|---|
| With $\sqrt{N/r}$ | {{D_WITH_CORR}} |
| Without | {{D_WITHOUT_CORR}} |

**Result.** {{H3_RESULT}}

**Verdict.** {{H3_VERDICT}}

## 8.2 Why Does GPT-2 Show No Change?

GPT-2 is causal; the code takes the decoder path:

```python
if self.is_decoder and not is_cross_attention:
    # Standard causal attention
    ...
else:
    # Galerkin path (never reached for GPT-2)
```

The decoder path is **identical to the baseline** (plus LoRA at rank 16). 
Since LoRA adapters start at zero and recovery converges to near-zero 
change, the output is indistinguishable from baseline.

**Result.** GPT-2 result is a **no-op**, not a success.

## 8.3 Why Does RoBERTa Collapse?

RoBERTa's accuracy ({{R_ACC}}%) is at chance level (majority class 50.92%). 
This indicates:

1. **Forward pass broken:** Model output is nearly constant.
2. **Cascading error:** 12 layers of Galerkin-projected attention compound 
   approximation error.
3. **Loss oscillation:** `{{R_MIN_LOSS}}` to `{{R_MAX_LOSS}}` in recovery.

**Hypothesis.** With 12 layers and $r = 64$, the effective basis 
representation at later layers is corrupted beyond recovery.

## 8.4 Attention Pattern Analysis

**Figure 8.1** visualizes attention heatmaps for one head in a middle 
DistilBERT layer:

- **Baseline:** Sharp, content-dependent peaks (attention sinks + local)
- **Galerkin:** Smoothed, global pattern (DCT projections average out peaks)

This confirms H2: the DCT basis acts as a **low-pass filter** on attention, 
removing the sharp structure that transformers rely on.

## 8.5 Singular Value Analysis

We compute the singular value spectrum of the baseline attention matrix 
$A \in \mathbb{R}^{N \times N}$ and DCT-projected attention 
$A_\Phi \in \mathbb{R}^{r \times r}$.

| Metric | Baseline | Galerkin |
|---|---|---|
| Effective rank | {{BASE_EFF_RANK}} | {{GAL_EFF_RANK}} |
| Top-1 energy | {{BASE_TOP1}} | {{GAL_TOP1}} |
| Entropy | {{BASE_ENTROPY}} | {{GAL_ENTROPY}} |

**Observation.** {{SV_OBS}}

## 8.6 Failure Mode Taxonomy

| Failure Mode | Symptom | Root Cause |
|---|---|---|
| **Encoder collapse** | Acc ≈ 50% | Fixed basis cannot adapt |
| **Encoder degrade** | -20% accuracy | Partial basis loss |
| **Decoder no-op** | Δ = 0% | Fallback path unmodified |
| **Loss oscillation** | Recovery instability | Approximation error |

## 8.7 Comparison with DCT-Based Attention

**FNet** (Lee-Thorp et al., 2021) replaces attention entirely with DCT 
and achieves ~85% on SST-2. Our method **projects** attention through DCT, 
preserving the softmax operation. Our result ({{D_ACC}}% on DistilBERT) is 
worse than FNet's, suggesting that:

1. Partial DCT filtering is **worse** than full DCT replacement.
2. Softmax on DCT-projected features produces uninformative attention.

## 8.8 Why the Fixed Basis Fails

The DCT basis $\Phi$ is a **data-independent, position-invariant** basis. 
Real attention is:

- **Data-dependent:** Different inputs produce different attention.
- **Position-sensitive:** Attention sinks at token 0, local attention at 
  edges, global attention in the middle.
- **Content-selective:** Some tokens are "heavy hitters" that receive 
  most attention.

A fixed DCT basis cannot encode any of these properties. Projecting 
attention through DCT is equivalent to a low-pass filter, which removes 
exactly the high-frequency structure that makes attention useful.

## 8.9 Summary of Insights

1. **Fixed DCT basis is fundamentally limited** for attention.
2. **Encoder models rely on sharp, content-dependent attention** that DCT 
   destroys.
3. **Decoder fallback is a no-op** — the module is ineffective for GPT-2.
4. **Rank increase does not help** — failure is not truncation.
5. **Galerkin Attention is not recommended** for any tested use case.

---