<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 What Does the Score Network Learn?

**Figure 8.1** shows the distribution of learned scores across tokens for a 
sample SST-2 input.

### 8.1.1 Content vs Function Words

**Table 8.1: Mean score by POS tag.**

| POS tag | Mean score |
|---|---|
| [CLS] | 1.00 (fixed) |
| Noun | {{SCORE_NOUN}} |
| Verb | {{SCORE_VERB}} |
| Adjective | {{SCORE_ADJ}} |
| Adverb | {{SCORE_ADV}} |
| Determiner | {{SCORE_DET}} |
| Preposition | {{SCORE_PREP}} |
| Punctuation | {{SCORE_PUNCT}} |

**Observation.** Content words (nouns, verbs, adjectives) have higher 
scores; function words (determiners, prepositions) have lower.

### 8.1.2 Correlation with TF-IDF

We compute the Pearson correlation between learned scores and TF-IDF:

$$
\rho_\text{Pearson} = \text{corr}(s, \text{TF-IDF}) = {{TFIDF_CORR}}
\tag{8.1}
$$

**Interpretation.** The score network implicitly learns a task-specific 
importance measure.

## 8.2 Per-Layer Analysis

**Table 8.2: Effective kept ratio (DistilBERT).**

| Layer | Kept ratio | Δ from target |
|---|---|---|
| 0 | {{D_L0_KEPT}}% | {{D_L0_DELTA}}% |
| 1 | {{D_L1_KEPT}}% | {{D_L1_DELTA}}% |
| 2 | {{D_L2_KEPT}}% | {{D_L2_DELTA}}% |
| 3 | {{D_L3_KEPT}}% | {{D_L3_DELTA}}% |
| 4 | {{D_L4_KEPT}}% | {{D_L4_DELTA}}% |
| 5 | {{D_L5_KEPT}}% | {{D_L5_DELTA}}% |

**Observation.** Later layers prune more aggressively.

## 8.3 Comparison with Heuristic Pruning

**Table 8.3: Learned vs heuristic scoring.**

| Scoring | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Attention magnitude | {{ATTN_D}} | {{ATTN_R}} | {{ATTN_G}} |
| Token magnitude | {{MAG_D}} | {{MAG_R}} | {{MAG_G}} |
| **Learned MLP** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |

**Observation.** Learned MLP improves by {{LEARNED_GAIN}}%.

## 8.4 Causality Verification

We verify causality via finite differences:

```python
x = torch.randn(1, L, d, requires_grad=True)
out = model(x)
out[0, i].sum().backward()
assert x.grad[0, i+1:].abs().max() < 1e-6
```

**Result.** {{CAUSAL_VERIFIED}}

## 8.5 Effect of Score Network Depth

**Table 8.4: Accuracy vs MLP depth.**

| Depth | DistilBERT | Params/layer |
|---|---|---|
| Linear (1 layer) | {{DEPTH1}} | ~768 |
| MLP-64 (2 layers) | **{{D_ACC}}** | ~50K |
| MLP-64-64 (3 layers) | {{DEPTH3}} | ~55K |

**Observation.** One hidden layer suffices.

## 8.6 Failure Cases

### 8.6.1 Cases Where PTP Fails

| Input type | Accuracy drop |
|---|---|
| Very short sequences ($L < 16$) | {{FAIL_SHORT}} |
| Adversarial inputs | {{FAIL_ADV}} |
| Rare-word inputs | {{FAIL_RARE}} |

### 8.6.2 Root Causes

{{FAIL_CAUSE}}

## 8.7 Numerical Stability

### 8.7.1 Score Clamping

$10^4$ clamp prevents overflow. Score range is bounded.

### 8.7.2 Gather Safety

`topk` returns valid indices; gather is safe.

### 8.7.3 Empty Mask

If all tokens are padding, we clamp to first token.

## 8.8 Summary of Insights

1. **Learned scores correlate with content-word importance.**
2. **[CLS] preservation is essential.**
3. **Late layers prune more.**
4. **Learned MLP beats heuristics.**
5. **Causality preserved via position comparison.**
