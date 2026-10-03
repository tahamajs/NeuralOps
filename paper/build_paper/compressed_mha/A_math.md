# Appendix A. Mathematical Derivations

## A.1 Joint QKV SVD

Given $W_Q, W_K, W_V$, stack:

$$
W_{QKV} = \begin{pmatrix} W_Q \\ W_K \\ W_V \end{pmatrix} \in \mathbb{R}^{(H d_h + 2 H_{kv} d_h) \times d}
\tag{A.1}
$$

Full SVD:

$$
W_{QKV} = U_{QKV} \Sigma_{QKV} V_{QKV}^\top
\tag{A.2}
$$

Truncate to rank $r$:

$$
W_{QKV} \approx U_r \Sigma_r V_r^\top
\tag{A.3}
$$

with $\|W_{QKV} - U_r \Sigma_r V_r^\top\|_F = \sqrt{\sum_{i > r} \sigma_i^2}$.

## A.2 Symmetric Factorization

Distribute $\Sigma^{1/2}$:

$$
U_r' = U_{:, :r} \Sigma_{:r, :r}^{1/2}, \quad
V_r' = \Sigma_{:r, :r}^{1/2} V_{:r, :}^\top
\tag{A.4}
$$

Store:
- $V_{QKV, r} = V_r'$ (shape $r \times d$)
- $S_{QKV, r} = \text{diag}(\Sigma_r^{1/2})$ (we store sqrt)
- $U_{QKV, r} = U_r'$ (shape $(H d_h + 2 H_{kv} d_h) \times r$)

**Forward:**

$$
\text{QKV} = \left((X V_{QKV, r}^\top) \cdot S_{QKV, r}\right) U_{QKV, r}^\top
\tag{A.5}
$$

## A.3 Parameter Count

| Factor | Shape | Params |
|---|---|---|
| $V_{QKV, r}$ | $r \times d$ | $r d$ |
| $S_{QKV, r}$ | $r$ | $r$ |
| $U_{QKV, r}$ | $(H d_h + 2 H_{kv} d_h) \times r$ | $r (H d_h + 2 H_{kv} d_h)$ |

For $H = H_{kv}$, $d_h = d/H$:
$$
\text{Total} = r d + r + r (d + 2 d_h) = r(2d + 2 d_h) + r
\tag{A.6}
$$

## A.4 Comparison with Independent SVD

Independent:
$$
\text{Params}_\text{indep} = 2 r d \cdot 3 = 6 r d
\tag{A.7}
$$

Joint:
$$
\text{Params}_\text{joint} = 2 r (d + d_h) + r
\tag{A.8}
$$

For $d = 768$, $d_h = 64$:
- Independent: $6 \cdot 384 \cdot 768 = 1,769,472$
- Joint: $2 \cdot 384 \cdot 832 + 384 = 639,360$

**Joint saves ~64%**.

## A.5 Output SVD

For $W_O \in \mathbb{R}^{d \times H d_h}$:

$$
W_O = U_O \Sigma_O V_O^\top
\tag{A.9}
$$

Truncate to $r_{out}$:

$$
W_O \approx U_{O, r} \Sigma_{O, r} V_{O, r}^\top
\tag{A.10}
$$

## A.6 LoRA on Both Branches

QKV LoRA:

$$
\text{QKV}_\text{LoRA} = \frac{\alpha}{r_L} B_{QKV} A_{QKV} X
\tag{A.11}
$$

Output LoRA:

$$
\text{Out}_\text{LoRA} = \frac{\alpha}{r_L} B_O A_O \text{Context}
\tag{A.12}
$$

Both have $B = 0$ at initialization.

## A.7 Reconstruction Error Bound

By Eckart-Young:

$$
\|W_{QKV} - W_{QKV, r}\|_F \leq \sqrt{\sum_{i > r} \sigma_i^2}
\tag{A.13}
$$

The energy fraction:

$$
E(r) = \frac{\sum_{i \leq r} \sigma_i^2}{\sum_i \sigma_i^2}
\tag{A.14}
$$

We can choose $r$ such that $E(r) \geq \eta$.

## A.8 Causality

Standard triangular mask before softmax:

$$
\text{mask}_{ij} = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}
\tag{A.15}
$$

## A.9 Nested Encoder Correctness

**Claim.** For nested encoders, skipping the output projection yields correct 
attention output.

**Proof.** The nested outer layer applies the output projection. Applying it 
in our module would result in $W_O \cdot (\text{Attn} \cdot W_O) \neq W_O \cdot \text{Attn}$, 
a double projection. Skipping our $W_O$ gives the correct 
$\text{Attn} \cdot W_O$ at the outer level. $\square$

---