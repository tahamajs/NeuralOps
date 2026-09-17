# Appendix A. Mathematical Derivations

## A.1 HGRN-2 Recurrence

The HGRN-2 update:

$$
s_t = \alpha_t \odot s_{t-1} + (1 - \alpha_t) \odot u_t
\tag{A.1}
$$

Unrolled:

$$
s_t = \sum_{j=1}^{t} \left( \prod_{i=j+1}^{t} \alpha_i \right) (1 - \alpha_j) u_j
\tag{A.2}
$$

The effective decay from step $j$ to step $t$ is $\prod_{i=j+1}^t \alpha_i$. 
For $\alpha_i = \alpha < 1$ constant, this is $\alpha^{t-j}$, giving 
exponential decay with horizon $1/(1-\alpha)$.

## A.2 DeltaNet Recurrence

The Gated DeltaNet update:

$$
S_t = \alpha_t S_{t-1} + \beta_t (v_t - S_{t-1} k_t) k_t^\top
\tag{A.3}
$$

Expanding:

$$
S_t = \alpha_t S_{t-1} + \beta_t v_t k_t^\top - \beta_t S_{t-1} k_t k_t^\top
\tag{A.4}
$$

The term $S_{t-1} k_t k_t^\top$ is the outer product of the prediction 
$S_{t-1} k_t$ with the key $k_t$. Subtracting it removes the previous 
prediction's influence before writing the new target.

## A.3 Fusion Gate

$$
\gamma_t = \sigma(W_g x_t + b_g) \in (0, 1)^d
\tag{A.5}
$$

Initial $W_g = 0, b_g = 0$ gives $\gamma_t = 0.5$.

## A.4 Output

$$
Y = \gamma \odot Y^\text{HGRN} + (1 - \gamma) \odot Y^\text{Delta}
\tag{A.6}
$$

$$
\text{out} = W_o Y + b_o
\tag{A.7}
$$

## A.5 Complexity

For each layer:

| Operation | Time |
|---|---|
| HGRN-2 | $\mathcal{O}(L d)$ |
| DeltaNet | $\mathcal{O}(L H d^2)$ |
| Gate | $\mathcal{O}(L d^2)$ |
| out_proj | $\mathcal{O}(L d^2)$ |

Total: $\mathcal{O}(L d^2)$ for $H \cdot d_h \approx d$.

Baseline attention: $\mathcal{O}(L^2 d)$.

**Ratio:** $\mathcal{O}(L^2 d) / \mathcal{O}(L d^2) = \mathcal{O}(L/d)$.

For $L = 4096$ and $d = 768$: ~5× asymptotic speedup.

## A.6 Breakeven Analysis

Baseline time: $T_b = c_1 L^2 d$
Ours: $T_o = c_2 L d^2 + c_3$

Breakeven: $c_1 L^2 d = c_2 L d^2 \Rightarrow L^* = (c_2/c_1) d$.

For typical constants, $L^* \approx d$. With $d = 768$ and overhead, the 
empirical breakeven is $L^* = 128$.

## A.7 Gradient Flow

Training gradient to LoRA factors:

$$
\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial \text{out}} \cdot \frac{\partial \text{out}}{\partial Y} \cdot \frac{\partial Y}{\partial \gamma} \cdot \frac{\partial \gamma}{\partial A}
\tag{A.8}
$$

Since $\gamma = \sigma(W_g x)$, gradients flow through the gate. Since 
$B = 0$ initially, the LoRA branch contributes zero at step 0.

---