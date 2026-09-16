# 3. Background

## 3.1 Linear Recurrent Attention

Given input $X \in \mathbb{R}^{B \times L \times d}$, a linear recurrent 
attention maintains a state $S_t$ updated at each timestep:

$$
S_t = f(S_{t-1}, x_t), \quad
y_t = g(S_t, x_t)
\tag{3.1}
$$

The output $y_t$ depends on $S_t$, which is a **fixed-size** summary of the 
past. This gives $\mathcal{O}(L)$ complexity in time and memory.

## 3.2 HGRN-2

HGRN-2 (Qin et al., 2024) uses an element-wise recurrence:

$$
s_t = \alpha_t \odot s_{t-1} + (1 - \alpha_t) \odot u_t
\tag{3.2}
$$

where:

- $u_t = \text{SiLU}(W_u x_t)$ is the candidate update.
- $\alpha_t = \sigma(W_\alpha x_t)$ is the **bounded forget gate** in $(0, 1)$.

**Interpretation.** The state $s_t \in \mathbb{R}^d$ is a vector where each 
element forgets independently. This is efficient and stable, but lacks 
cross-feature expressivity.

## 3.3 DeltaNet

The Delta rule (Widrow & Hoff, 1960) updates a matrix memory by the 
prediction error:

$$
S_t = S_{t-1} + \beta_t (v_t - S_{t-1} k_t) k_t^\top
\tag{3.3}
$$

where:

- $k_t, v_t \in \mathbb{R}^d$ are key and value.
- $\beta_t \in (0, 1)$ is a scalar update gate.
- $S_t \in \mathbb{R}^{d \times d}$ is the matrix state.

**Interpretation.** The term $(v_t - S_{t-1} k_t)$ is the error between the 
target $v_t$ and the current prediction $S_{t-1} k_t$. The update writes this 
error into the memory, keyed by $k_t$.

**Gating.** Gated DeltaNet adds a forget gate:

$$
S_t = \alpha_t S_{t-1} + \beta_t (v_t - S_{t-1} k_t) k_t^\top
\tag{3.4}
$$

This allows the memory to decay when the input distribution shifts.

## 3.4 Gating and Mixture

Gating is a classical technique for combining multiple pathways:

$$
y = \gamma \odot y_1 + (1 - \gamma) \odot y_2, \quad
\gamma = \sigma(W_g x)
\tag{3.5}
$$

In Mixture-of-Experts (Shazeer et al., 2017), $\gamma$ is a routing weight. 
In our case, $\gamma$ is a **per-token scalar** that decides how much to 
trust each branch.

## 3.5 LoRA Recovery

After replacing attention, we freeze the base weights and train only LoRA 
adapters (Hu et al., 2021) plus the gate:

$$
\Delta W = \frac{\alpha}{r} B A
\tag{3.6}
$$

with $B = 0$ at initialization.

## 3.6 Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden dimension |
| $H$ | Heads |
| $s_t$ | HGRN-2 state |
| $S_t$ | DeltaNet state |
| $\alpha_t$ | Forget gate |
| $\beta_t$ | Update gate |
| $\gamma$ | Fusion gate |
| $W_u, W_\alpha, W_g$ | Learned projections |

---