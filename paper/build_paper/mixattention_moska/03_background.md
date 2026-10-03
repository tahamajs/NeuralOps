<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Scaled Dot-Product Attention

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)V
\tag{3.1}
$$

**Cost analysis.** The score matrix $QK^\top$ is $O(B H L^2)$. The value 
matmul $AV$ is $O(B H L^2 d_h)$.

## 3.2 Token Pruning

Replace $K, V$ with subsets $K_{\mathcal{S}}, V_{\mathcal{S}}$:

$$
\text{Attn}'(Q, K, V) = \text{softmax}\!\left(\frac{QK_{\mathcal{S}}^\top}{\sqrt{d_h}}\right)V_{\mathcal{S}}
\tag{3.2}
$$

**Cost.** $O(B H L \cdot |\mathcal{S}|)$ for scores and $O(B H L |\mathcal{S}| d_h)$ 
for the value matmul. With $|\mathcal{S}| = rL$, cost reduces by factor $r$.

## 3.3 Variance as Importance

For a token's key vector $k \in \mathbb{R}^{d_h}$:

$$
\text{Var}(k) = \frac{1}{d_h} \sum_{j=1}^{d_h} (k_j - \bar{k})^2
\tag{3.3}
$$

**Intuition.** Tokens whose keys have high variance have distinctive 
dimensional activations. They are likely to be attended to because their 
key inner-products with queries will span a wider range.

**Counter-argument.** This is a **heuristic**. Tokens with low-variance 
keys may still be highly attended if all queries align with them. Conversely, 
high-variance keys may be noisy and unattended. The signal is imperfect.

## 3.4 Attention Sinks

**Empirical observation.** The first token receives disproportionate 
attention in most transformer layers, even without any specific pretraining 
objective targeting this.

**Consequence.** Pruning the first token is catastrophic. FASA protects it:

```python
importance[:, 0] = 1e9
```

## 3.5 LoRA

LoRA (Hu et al., 2021):

$$
W_{\text{eff}} = W_{\text{frozen}} + \frac{\alpha}{r_L} BA
\tag{3.4}
$$

Used for recovery after injection.

## 3.6 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d_h$ | Head dimension |
| $H$ | Number of heads |
| $r$ | Keep ratio |
| $\mathcal{S}$ | Selected token subset |
| $\text{Var}(k)$ | Variance of key vector |
| $r_L$ | LoRA rank |
</parameter>

---