# Appendix A. Mathematical Derivations

## A.1 Expert-Choice Routing

Given input $X \in \mathbb{R}^{T \times d}$ with $T = B \cdot L$:

$$
\mathbf{r} = W_g \cdot \text{LayerNorm}(X) \in \mathbb{R}^{T \times E}
\tag{A.1}
$$

Softmax over tokens (axis 0):

$$
\mathbf{s}_{t, e} = \frac{\exp(\mathbf{r}_{t, e})}{\sum_{t'=1}^{T} \exp(\mathbf{r}_{t', e})}
\tag{A.2}
$$

For each expert $e$:

$$
\text{Top-}k_e = \text{TopK}(\mathbf{s}[:, e], k), \quad k = \left\lceil \frac{T}{E} \cdot \text{CF} \right\rceil
\tag{A.3}
$$

**Perfect balance.** $|\text{Top-}k_e| = k$ for all $e$.

## A.2 Expert Forward

$$
\mathbf{y}_e = \text{Expert}_e(X[\text{Top-}k_e])
\tag{A.4}
$$

$$
\text{Expert}_e(\mathbf{x}) = W_2^{(e)} \text{GELU}(W_1^{(e)} \mathbf{x})
\tag{A.5}
$$

## A.3 Aggregation

$$
\mathbf{y}_{\text{out}}[t] = \sum_{e : t \in \text{Top-}k_e} \mathbf{s}_{t, e} \cdot \mathbf{y}_e[\text{position of } t \text{ in } \text{Top-}k_e]
\tag{A.6}
$$

## A.4 Load Balance Proof

**Claim.** Each expert processes exactly $k$ tokens.

**Proof.** By construction, expert $e$ selects the top-$k$ tokens of 
$\mathbf{s}[:, e]$. The `torch.topk` operation returns exactly $k$ 
indices. $\square$

## A.5 Zero-Init Guarantee

**Claim.** If $W_2^{(e)} = 0$ and $b_2^{(e)} = 0$ for all $e$, then 
$\text{MoE\_out} = 0$.

**Proof.** Each expert output: $\text{Expert}_e(\mathbf{x}) = W_2^{(e)} 
\text{GELU}(\cdot) + b_2^{(e)} = 0$. So $\text{MoE\_out} = 0$. $\square$

## A.6 Gate Saturation

$$
\gamma = \sigma(0.1) = \frac{1}{1 + e^{4}} \approx 0.8
\tag{A.7}
$$

## A.7 Parameter Count

For $E$ experts with hidden size $2d$:

$$
\text{params}_\text{expert} = 2 d^2 + 2 d^2 = 4 d^2
\tag{A.8}
$$

$$
\text{params}_\text{MoE} = E \cdot 4 d^2 + E d
\tag{A.9}
$$

For $d = 768$, $E = 8$: ~$18.9$M per layer.

## A.8 Complexity

| Operation | Time | Memory |
|---|---|---|
| Router | $\mathcal{O}(T E d)$ | $\mathcal{O}(T E)$ |
| Softmax | $\mathcal{O}(T E)$ | $\mathcal{O}(T E)$ |
| TopK | $\mathcal{O}(E \cdot T \log k)$ | $\mathcal{O}(E k)$ |
| Expert forward | $\mathcal{O}(T d \cdot d_\text{hidden})$ | $\mathcal{O}(k \cdot d_\text{hidden})$ |
| Aggregation | $\mathcal{O}(E k d)$ | $\mathcal{O}(T d)$ |
| **Total** | $\mathcal{O}(T E d)$ | $\mathcal{O}(T E + T d)$ |

**Comparison to token-choice:** Same asymptotics; expert-choice is simpler 
(no auxiliary loss).

---