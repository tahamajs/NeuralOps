<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Mathematical derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Sliding-Window Mask

For position $i$ and key position $j$:

$$
M_{ij} = \mathbf{1}\left[|i - j| \leq W\right] \cdot \mathbf{1}\left[\text{valid}_j\right]
\tag{A.1}
$$

where $\text{valid}_j \in \{0, 1\}$ is the padding mask.

## A.2 Fusion Formula

$$
\text{fused}_t = g_t \cdot \text{short}_t + (1 - g_t) \cdot \text{long}_t
\tag{A.2}
$$

where:
- $g_t = \text{mean}_h \sigma(W_g x_t + b_g)_h \in [0, 1]$
- $\text{short}_t \in \mathbb{R}^{H d_h}$ (concatenation of head outputs)
- $\text{long}_t \in \mathbb{R}^{d}$ (from linear projection)

**Type mismatch.** $\text{short}_t$ is $H d_h = d$ dimensional (for MHSA 
with $H d_h = d$). $\text{long}_t$ is $d$ dimensional. So the fusion is 
valid when $d_h H = d$.

## A.3 Preservation Gate

$$
Y = \alpha \cdot \text{out} + (1 - \alpha) \cdot x
\tag{A.3}
$$

with $\alpha = \sigma(w_\alpha)$, $w_\alpha \in \mathbb{R}$.

**Initialization.** $w_\alpha = 2.0$ gives $\alpha = \sigma(2.0) = 0.88$.

**Identity at initialization.** With $\alpha = 0.88$, the block is 88% 
identity-like. This is **more conservative** than ReZero ($\alpha = 0$) 
which is fully identity at init.

## A.4 The Long-Memory Branch

The "GatedDeltaNetLayer" computes:

$$
\text{long}_t = W_m x_t + b_m
\tag{A.4}
$$

with $W_m \in \mathbb{R}^{d \times d}$, $b_m \in \mathbb{R}^d$.

**Not recurrent.** Since $\text{long}_t$ depends only on $x_t$, there is no 
long-range memory.

## A.5 Parameter Count

Per layer:

| Component | Params |
|---|---|
| $W_m$ (long-memory) | $d^2 + d$ |
| $W_g$ (gate) | $d H + H$ |
| LoRA (4 proj) | $4 \cdot 2 r d$ |
| $\alpha$ (preservation) | 1 |
| `output_scale` | 1 |

For DistilBERT ($d = 768$, $H = 12$, $r = 16$):
- $W_m$: 590,592
- $W_g$: 9,228
- LoRA: 98,304
- Others: 2
- **Total: ~698,126**

## A.6 FLOP Analysis

| Operation | FLOPs |
|---|---|
| Q/K/V projection | $3 L d^2$ |
| Sliding window scores | $L W d$ |
| Softmax | $L W$ |
| Attention output | $L W d$ |
| Long-memory linear | $L d^2$ |
| Gate | $L d H$ |
| Fusion | $L d$ |
| Output projection | $L d^2$ |
| Preservation gate | $L d$ |
| **Total** | $\sim 5 L d^2 + 2 L W d$ |

Compared to baseline MHSA: $4 L d^2 + L^2 d$.

For $L = 128$, $d = 768$: GoldFinch ~ $5 \cdot 128 \cdot 768^2 \approx 
377M$ FLOPs vs baseline $\approx 4 \cdot 128 \cdot 768^2 + 128^2 \cdot 768 
\approx 300M + 12M = 312M$.

---