# Appendix A. Mathematical Derivations

## A.1 Recurrence Unrolling

The decay recurrence:

$$
S_t = \lambda_t S_{t-1} + \phi(k_t) v_t^\top
\tag{A.1}
$$

With $S_0 = 0$:

$$
S_t = \sum_{j=1}^{t} \left( \prod_{i=j+1}^{t} \lambda_i \right) \phi(k_j) v_j^\top
\tag{A.2}
$$

**Decay coefficient:** $D_{t, j} = \exp(\sum_{i=j+1}^t \log \lambda_i)$.

## A.2 Chunked Scan Correctness

**Claim.** The chunked scan produces the same output as the direct 
recurrence.

**Proof sketch.** By induction on chunks.

**Base case.** Chunk 0: no previous state. The intra-chunk computation 
matches the recurrence.

**Inductive step.** Assume chunk $c-1$ left state $(S_{c-1}, z_{c-1})$ 
matching the direct recurrence at the chunk boundary. Then:

- **Intra-chunk** computation uses the local cumsum, matching the recurrence 
  within the chunk.
- **Inter-chunk** contributions use the decayed state $D_i^\text{inter} = 
  \exp(\Lambda_i)$ applied to $S_{c-1}$, which is exactly the decay from the 
  boundary to position $i$.
- **State update** produces $S_c$ via (4.16), which matches the recurrence 
  at the chunk boundary.

$\square$

## A.3 ELU+1 Positivity

$$
\phi(x) = \text{elu}(x) + 1 > 0 \quad \forall x \in \mathbb{R}
\tag{A.3}
$$

**Proof.** $\text{elu}(x) \in (-1, \infty)$, so $\phi(x) \in (0, \infty)$. 
$\square$

## A.4 Data-Dependent Decay

$$
\lambda_{h,t} = \sigma(W_g x_t + b_g + \ell_h)
\tag{A.4}
$$

Initialization: $W_g = 0, b_g = 0$, $\ell_h \in [-0.5, 0.0]$.

**Head diversity.** $\lambda_{h} = \sigma(\ell_h)$ ranges:
- $\lambda_0 = \sigma(-0.5) \approx 0.38$
- $\lambda_{H-1} = \sigma(0.0) \approx 0.50$

Effective horizons:
- Head 0: $1/(1-\lambda_0) \approx 1.6$
- Head $H-1$: $1/(1-\lambda_{H-1}) \approx 2.0$

## A.5 Complexity

For a single layer:

| Component | Time | Memory |
|---|---|---|
| Projections | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| Cumsum | $\mathcal{O}(L)$ | $\mathcal{O}(L)$ |
| Intra-chunk | $\mathcal{O}(L \cdot C)$ | $\mathcal{O}(L \cdot C)$ |
| Inter-chunk | $\mathcal{O}(L \cdot d_h)$ | $\mathcal{O}(d_h^2)$ |
| State update | $\mathcal{O}(L \cdot d_h^2 / C)$ | $\mathcal{O}(d_h^2)$ |

**Total memory:** $\mathcal{O}(L d + L C + d_h^2)$.

For $C = 32$ and $d_h = 64$: state memory $= 16$ KB per layer.

## A.6 Bidirectional Combination

$$
O = \frac{1}{2} (O^\text{fwd} + O^\text{bwd})
\tag{A.5}
$$

**Why 0.5?** Maintains output magnitude. Sum would double, average avoids it.

## A.7 Cross-Attention

For encoder-decoder attention, K, V come from encoder states:

$$
K = H_\text{enc} W_K, \quad V = H_\text{enc} W_V
\tag{A.6}
$$

Q from decoder:

$$
Q = H_\text{dec} W_Q
\tag{A.7}
$$

Same chunked scan applies.

---