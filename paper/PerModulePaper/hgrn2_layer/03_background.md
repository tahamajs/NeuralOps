# 3. Background

## 3.1 Linear Recurrence

A linear recurrent layer maintains a state $s_t \in \mathbb{R}^d$ updated 
per timestep:

$$
s_t = f_t \odot s_{t-1} + g_t
\tag{3.1}
$$

where $f_t \in \mathbb{R}^d$ is the forget (decay) vector and $g_t \in 
\mathbb{R}^d$ is the input. This is a **first-order linear recurrence**.

**Complexity.** Time is $\mathcal{O}(L d)$; memory is $\mathcal{O}(d)$ (or 
$\mathcal{O}(L d)$ for the outputs). Compare to attention's $\mathcal{O}(L^2 d)$.

## 3.2 HGRN-2 Recurrence

HGRN-2 (Qin et al., 2024) parameterizes (3.1) as:

$$
f_t = \exp(-\text{softplus}(W_f x_t + b_f))
\tag{3.2}
$$

$$
i_t = \sigma(W_i x_t + b_i)
\tag{3.3}
$$

$$
v_t = W_v x_t
\tag{3.4}
$$

$$
s_t = f_t \odot s_{t-1} + (1 - f_t) \odot i_t \odot v_t
\tag{3.5}
$$

**Bounded forget.** Since $\text{softplus}(u) > 0$, we have $f_t \in (0, 1)$. 
We clamp further to $[10^{-3}, 1 - 10^{-4}]$ to prevent numerical extremes.

**Input modulation.** The factor $(1 - f_t)$ scales the input so that when 
$f_t$ is small (forget), the input is large (write). This avoids sign-flipping 
in the recurrent state.

## 3.3 Parallel Chunked Scan

The recurrence (3.5) is **associative**: it can be computed by a prefix scan. 
For a chunk of length $C$, define the cumulative log-forget:

$$
A_k = \sum_{j=1}^{k} \log f_j
\tag{3.6}
$$

Then within the chunk:

$$
h_k = \sum_{j=1}^{k} \exp(A_k - A_j) \cdot g_j
\tag{3.7}
$$

where $g_j = (1 - f_j) \odot i_j \odot v_j$.

For chunks, we carry the state $h_\text{prev}$ from the previous chunk and 
combine:

$$
h_k^\text{chunk} = h_\text{prev} \cdot \exp(A_k) + \text{intra-chunk sum}
\tag{3.8}
$$

This reduces the sequential dependency from $L$ steps to $L / C$ steps 
(one per chunk), and inside each chunk the computation is parallel.

## 3.4 Bidirectional Recurrence

For encoders, we run two scans:

- Forward: $h_t^\text{fwd} = \text{scan}(f, g)$ over $t = 1, \ldots, L$.
- Backward: $h_t^\text{bwd} = \text{scan}(\text{flip}(f), \text{flip}(g))$ over 
  $t = L, \ldots, 1$.

The output combines both:

$$
h_t = h_t^\text{fwd} + h_t^\text{bwd}
\tag{3.9}
$$

This gives each position access to both past and future context.

## 3.5 LoRA Recovery

We freeze base weights and train only LoRA adapters:

$$
\Delta W = \frac{\alpha}{r} B A
\tag{3.10}
$$

with $B = 0$ initially.

## 3.6 Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden dim |
| $x_t$ | Input at position $t$ |
| $s_t$ | State |
| $f_t$ | Forget gate |
| $i_t$ | Input gate |
| $v_t$ | Value |
| $g_t$ | Input contribution $(1-f_t) i_t v_t$ |
| $A_k$ | Cumulative log-forget |
| $C$ | Chunk size |

---