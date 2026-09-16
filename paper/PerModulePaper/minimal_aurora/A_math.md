# Appendix A. Mathematical Derivations

## A.1 Energy Bound

**Claim.** With energy threshold $\varepsilon$:

$$
\|W - W_{r(\varepsilon)}\|_F \leq \sqrt{1 - \varepsilon} \cdot \|W\|_F
$$

**Proof.**

$$
\|W\|_F^2 = \sum_{i=1}^{\min(m,n)} \sigma_i^2
$$

$$
\|W - W_r\|_F^2 = \sum_{i > r} \sigma_i^2 = (1 - E(r)) \|W\|_F^2
$$

With $E(r) \geq \varepsilon$:

$$
\|W - W_r\|_F^2 \leq (1 - \varepsilon) \|W\|_F^2 \implies \|W - W_r\|_F \leq \sqrt{1 - \varepsilon} \|W\|_F \quad \square
$$

## A.2 Rank Selection

**Claim.** The energy-selected rank satisfies:

$$
r(\varepsilon) = \min\{r : E(r) \geq \varepsilon\}
$$

**Implementation:**

```python
cum_energy = torch.cumsum(S**2, dim=0) / (S**2).sum()
r = int((cum_energy < target_energy).sum().item()) + 1
```

**Proof.** The smallest $r$ with $E(r) \geq \varepsilon$ is the number of 
$i$ with $E(i) < \varepsilon$, plus one. $\square$

## A.3 Parameter Count

**Claim.** Per projection:

$$
P_\text{MinimalAURORA} = r(d_\text{in} + d_\text{out}) + d_\text{out} + r_L(d_\text{in} + d_\text{out})
$$

**Proof.** Down: $r d_\text{in}$. Up: $r d_\text{out} + d_\text{out}$. LoRA: $r_L(d_\text{in} + d_\text{out})$. Sum. $\square$

## A.4 Causality

**Proof.** Position-wise projections + causal mask. $\square$

## A.5 Nested Encoder Correctness

**Claim.** With identity out_proj, RoBERTa attention output equals the 
original.

**Proof.** For RoBERTa, `attention.output.dense` applies $W_O$ outside 
`attention.self`. Setting MinimalAURORA's `out_proj` to identity means 
its contribution is $I \cdot \text{context} = \text{context}$. The outer 
`attention.output.dense` applies $W_O$. Combined: $W_O \cdot \text{context}$. 
Same as baseline. $\square$

---