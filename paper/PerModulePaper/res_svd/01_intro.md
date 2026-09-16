# 1. Introduction

## 1.1 Motivation

Low-rank factorization of transformer weights is a common compression 
primitive. For each projection $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, 
one replaces $W$ with a product of two smaller matrices, reducing parameter 
count from $d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out})$ 
for rank $r$.

**Two standard variants exist:**

1. **Pure SVD compression** (ASVD, AASVD): replace $W$ with a rank-$r$ SVD 
   approximation, then fine-tune with LoRA adapters on top. The LoRA is 
   initialized to zero.
2. **Pure PEFT** (LoRA): keep $W$ dense, add a rank-$r$ trainable adapter. 
   No compression.

**Res-SVD bridges these.** The base is a low-rank SVD approximation (compressed), 
and the residual is a trainable adapter (PEFT), but the residual is **not 
initialized to zero** — it is initialized from the **next singular directions** 
of $W$. This makes the combined approximation equal to the best rank-$(r_b + r_r)$ 
SVD at initialization, while keeping the base frozen for fine-tuning.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_\ell$ attention layers, 
we seek a compressed model $\mathcal{M}'$ such that:

1. Each projection is stored as a sum of two low-rank factors:
   $$
   W \approx U_b V_b^\top + \alpha B A
   \tag{1.1}
   $$
   with $r_b + r_r \ll \min(d_\text{in}, d_\text{out})$.
2. Parameter count decreases by factor $\sim (d_\text{in} d_\text{out}) / ((r_b + r_r)(d_\text{in} + d_\text{out}))$.
3. The base $U_b V_b^\top$ is frozen; only $B, A$ are trained.
4. At initialization, $U_b V_b^\top + \alpha B A$ matches the best rank-$(r_b + r_r)$ 
   SVD of $W$.
5. Weight transfer works across four attention families.

## 1.3 Our Approach: Res-SVD Attention

Given a projection weight $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$:

**Step 1.** Compute the top-$r_b$ SVD:
$$
W_b = U_b \Sigma_b V_b^\top, \quad r_b \ll \min(d_\text{in}, d_\text{out})
\tag{1.2}
$$

**Step 2.** Compute the residual:
$$
R = W - W_b
\tag{1.3}
$$

**Step 3.** Take the top-$r_r$ SVD of the residual:
$$
R \approx U_r \Sigma_r V_r^\top
\tag{1.4}
$$

**Step 4.** Initialize the LoRA adapter with:
$$
B = U_r \Sigma_r^{1/2}, \quad A = \Sigma_r^{1/2} V_r^\top
\tag{1.5}
$$

**Step 5.** Store $U_b, V_b$ as frozen parameters and $B, A$ as trainable.

The forward pass computes:
$$
y = x (U_b V_b)^\top + \alpha x (B A)^\top + b
\tag{1.6}
$$

## 1.4 Contributions

- **A two-level SVD factorization.** Res-SVD combines frozen low-rank SVD 
  with a trainable residual, initialized from the second-order spectrum.
- **Spectral LoRA initialization.** The residual adapter is initialized 
  non-zero, unlike standard LoRA, ensuring the initial approximation is 
  optimal for the total rank.
- **Multi-architecture support.** Same code handles flat, nested, causal, 
  and GQA+RoPE attention.
- **Compatibility with RoBERTa.** Nested encoders are handled via 
  identity initialization of the output projection.
- **Honest reporting.** Res-SVD reduces parameter count. The residual 
  adapter is a small trainable addition.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on SVD, LoRA, 
and spectral initialization. Section 4 presents Res-SVD. Section 5 details 
multi-architecture integration. Section 6 gives setup. Section 7 reports 
results. Section 8 analyzes contributions. Section 9 provides guidance. 
Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $W$ | Projection weight matrix |
| $r_b$ | Base SVD rank |
| $r_r$ | Residual (LoRA) rank |
| $U_b, V_b$ | Base SVD factors (frozen) |
| $B, A$ | Residual LoRA factors (trainable) |
| $\alpha$ | Residual scaling |
| $R$ | Residual matrix $W - U_b V_b^\top$ |
| $d_\text{in}, d_\text{out}$ | Input/output dimensions |

---