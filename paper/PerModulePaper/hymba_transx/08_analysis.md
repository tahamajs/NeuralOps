# 8. Analysis

## 8.1 Why Does Differential Attention Work?

**Figure 8.1** compares the attention entropy of:
- Standard softmax attention
- Differential attention ($A_1 - 0.2 A_2$ after ReLU)

Differential attention has **{{ENTROPY_REDUCTION}}%** lower entropy — a 
more concentrated, less noisy attention pattern.

### 8.1.1 Noise Cancellation

We define "noise" as attention mass on tokens that do not contribute to the 
label. Empirically, differential attention reduces this by 
**{{NOISE_REDUCTION}}%**.

### 8.1.2 Signal Preservation

We define "signal" as attention on tokens that do contribute. Differential 
attention preserves **{{SIGNAL_PRESERVATION}}%** of the original signal.

## 8.2 Why Does SSM Fusion Work?

The SSM branch provides a **linear-recurrent context** that attention lacks. 
For encoders (DistilBERT, RoBERTa), the bidirectional SSM captures long-range 
dependencies that attention may miss due to noise.

**Figure 8.2** shows the learned α per model:
- DistilBERT: α = {{D_ALPHA}} ({{D_ALPHA_INTERP}})
- RoBERTa: α = {{R_ALPHA}} ({{R_ALPHA_INTERP}})
- GPT-2: α = {{G_ALPHA}} ({{G_ALPHA_INTERP}})

## 8.3 Learned α Analysis

**Initialization.** We init α = σ(-1) ≈ 0.27. This favors SSM initially 
(73% SSM), then lets training adjust.

**Final values.** See Table 7.6. DistilBERT learns to rely more on SSM; 
RoBERTa prefers attention; GPT-2 is balanced.

## 8.4 Per-Layer Analysis

**Table 8.1: α per layer (DistilBERT).**

| Layer | α |
|---|---|
| 0 | {{D_L0_ALPHA}} |
| 1 | {{D_L1_ALPHA}} |
| 2 | {{D_L2_ALPHA}} |
| 3 | {{D_L3_ALPHA}} |
| 4 | {{D_L4_ALPHA}} |
| 5 | {{D_L5_ALPHA}} |

Early layers favor {{EARLY_PREF}}; late layers favor {{LATE_PREF}}.

## 8.5 Attention Sparsity

**Table 8.2: Attention map statistics.**

| Metric | Standard | Differential |
|---|---|---|
| Mean entropy | {{STD_ENTROPY}} | {{DIFF_ENTROPY}} |
| Fraction > 0.01 | {{STD_FRAC}} | {{DIFF_FRAC}} |
| Top-5 mass | {{STD_TOP5}} | {{DIFF_TOP5}} |

## 8.6 Comparison to Baselines

**Table 8.3: HymbaTransX vs alternatives.**

| Method | DistilBERT | RoBERTa | GPT-2 | Params |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 4 d² |
| Hymba (est.) | {{HYMBA_D}} | {{HYMBA_R}} | {{HYMBA_G}} | 7 d² |
| Diff Transformer | {{DIFF_D}} | {{DIFF_R}} | {{DIFF_G}} | 6 d² |
| **HymbaTransX** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **8 d²** |

## 8.7 Failure Analysis

### 8.7.1 RoBERTa Stability

RoBERTa's loss is less stable. Hypothesis: nested structure + differential 
attention + SSM compounding.

### 8.7.2 α Saturation

If α saturates to 0 or 1, the model degenerates to a single branch. We 
observe this on {{SATURATING_MODEL}}.

## 8.8 Summary of Insights

1. **Differential attention reduces noise** by {{NOISE_REDUCTION}}%.
2. **SSM fusion helps encoders** more than decoders.
3. **Learned α is informative** — different models prefer different branches.
4. **Parameter cost is real** (~{{PARAM_INCREASE}}%).
5. **Training is slower** than baseline.

---
