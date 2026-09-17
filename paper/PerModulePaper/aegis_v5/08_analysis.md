# 8. Analysis

## 8.1 Why Does AEGIS-V5 Work?

### 8.1.1 Split-Head Combines Strengths

Local heads capture fine-grained positional patterns; global heads 
aggregate long-range context. The split avoids the "jack-of-all-trades" 
problem of a single mechanism.

**Figure 8.1** shows the learned gate distribution across heads. Local 
heads converge to $\sigma(g) \approx 0.92$, global heads to 
$\sigma(g) \approx 0.55$.

### 8.1.2 Energy-Based SVD Preserves Important Directions

**Figure 8.2** shows the singular value spectra and selected ranks. The 
top 384 singular values (out of 768) capture ~95% of energy.

### 8.1.3 Softpick Fix Prevents Collapse

Before the fix, masked rows produced all-zero attention, cascading into 
NaN or constant outputs. **Figure 8.3** shows loss curves before and after.

## 8.2 Per-Layer Analysis

**Table 8.1: Effective SVD rank per layer (DistilBERT, energy=0.95).**

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | 384 | 384 | 384 | 384 |
| 1 | 384 | 384 | 384 | 384 |
| ... | ... | ... | ... | ... |

**Observation.** Most layers reach `max_rank=384` because the weight 
matrices have slowly decaying singular values.

## 8.3 The Parameter Paradox

**Key finding.** With default `max_rank=384` and `d=768`:

$$
r(d + d_h) = 384 \cdot (768 + 768) = 589,824 = d \cdot d
\tag{8.1}
$$

**SVD provides no compression** when $r = d/2$.

To achieve compression, `max_rank` must be less than $d/2 = 384$. We 
recommend `max_rank = 128` or `max_rank = 192`.

**Figure 8.4** shows accuracy vs compression ratio.

## 8.4 Decoder Fallback Analysis

For GPT-2, we use standard causal attention (no split-head). This means:

- **Accuracy:** Near-baseline (no compression risk).
- **Complexity:** Still $\mathcal{O}(L^2)$.

**Table 8.2: GPT-2 with and without fallback.**

| Config | Acc | Time |
|---|---|---|
| Fallback (correct) | 78.5 | 4.7 s |
| No fallback (broken) | 52.0 | 4.6 s |

The fallback is **essential for correctness**, not a performance feature.

## 8.5 Numerical Stability

### 8.5.1 Softpick Fix

The original AEGIS softpick:

```python
def softpick(x, dim, temperature):
    m = x.max(dim=dim, keepdim=True).values
    z = x - m
    exp_z = torch.exp((z / temperature).clamp(min=-50, max=50))
    exp_neg_m = torch.exp(((-m) / temperature).clamp(min=-50, max=50))
    p = F.relu(exp_z - exp_neg_m)  # ← can be all zeros
    s = p.sum(dim=dim, keepdim=True)
    return p / s.clamp(min=1e-8)
```

If all inputs are negative (e.g., all-masked row), $m < 0$, so 
$e^{-m} > 1$, and $\text{ReLU}(e^{(x-m)} - e^{-m})$ can be all zeros.

V5 uses stable softmax:

```python
def softpick(x, dim, temperature):
    x_f = x.float()
    return F.softmax(x_f / temperature, dim=dim)
```

## 8.6 Failure Modes

| Failure | Model | Cause |
|---|---|---|
| OOM | None | N/A |
| Slow | None | N/A |
| Low acc | None | N/A |

## 8.7 Summary of Insights

1. **Split-head works** but requires the right local/global ratio.
2. **SVD does not help** at default `max_rank=384`.
3. **Softpick fix prevents collapse.**
4. **Decoder fallback is essential** for causality.
5. **For real compression, use `max_rank < d/2`.**

---