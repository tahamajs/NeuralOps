# 2. Related Work

## 2.1 Low-Rank Factorization

### 2.1.1 Classical SVD

Eckart-Young (1936): optimal rank-$r$ Frobenius approximation is truncated SVD. 
Res-SVD uses SVD twice: for base and residual.

### 2.1.2 Randomized SVD

Halko et al. (2011) — randomized algorithm for truncated SVD. Not needed 
for our layer sizes.

### 2.1.3 ASVD / AASVD / A3

Companion modules: use SVD once with optional activation weighting. Res-SVD 
uses SVD twice with spectral initialization of the residual.

## 2.2 Parameter-Efficient Fine-Tuning

### 2.2.1 LoRA

Hu et al. (2021) — standard LoRA initializes $B = 0$. This makes the 
initial approximation equal to the base weight. Res-SVD initializes 
$B \neq 0$ from the residual spectrum.

### 2.2.2 AdaLoRA

Zhang et al. (2023) — adaptive rank allocation. Not directly related.

### 2.2.3 DoRA

Liu et al. (2024) — weight-decomposed LoRA. Different decomposition.

### 2.2.4 PiSSA

Meng et al. (2024) — **PiSSA initializes LoRA from the top singular 
directions of the base weight**, making the LoRA "principal". Res-SVD is 
the **complementary** approach: base takes principal directions, residual 
takes the next ones.

## 2.3 Residual Decomposition

### 2.3.1 Two-Level SVD

Any full-rank matrix $W$ can be written as $W = W_b + R$ where $W_b$ is 
rank $r_b$. The residual $R$ has rank $\leq \min(d_\text{in}, d_\text{out})$ 
but effective rank $r_r$ if its spectrum decays.

### 2.3.2 Connection to Optimal Rank-$(r_b + r_r)$ SVD

As shown in Appendix A.1, Res-SVD with $(r_b, r_r)$ gives the same 
approximation as SVD of rank $r_b + r_r$.

## 2.4 Position of Our Work

| Aspect | Pure SVD | LoRA | PiSSA | Res-SVD |
|---|---|---|---|---|
| Base frozen | Yes | No (dense) | No | Yes |
| Base rank | $r$ | Full | Full | $r_b$ |
| Residual rank | 0 | $r$ | $r$ | $r_r$ |
| Residual init | N/A | Zero | Principal | **Spectral** |
| Compression | Yes | No | No | **Yes** |

## 2.5 Comparison to Companion Modules

| Module | Approach | Base | Residual |
|---|---|---|---|
| ASVD | One SVD | Rank $r$ | LoRA zero |
| AASVD | One SVD + covariance | Rank $r$ | LoRA zero |
| A3 | One SVD + Fisher | Rank $r$ | LoRA zero |
| SVD-Linear | SVD + linear attn | Rank $r$ | LoRA zero |
| **Res-SVD** | **Two-level SVD** | **Rank $r_b$** | **Spectral init** |

---