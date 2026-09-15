# Appendix A. Mathematical Derivations

## A.1 Online Softmax

Standard softmax:

$$
P_{ij} = \frac{e^{s_{ij}}}{\sum_k e^{s_{ik}}}
\tag{A.1}
$$

Numerically stable:

$$
P_{ij} = \frac{e^{s_{ij} - m_i}}{\sum_k e^{s_{ik} - m_i}}
\tag{A.2}
$$

where $m_i = \max_k s_{ik}$.

## A.2 Tiled Computation

Split key indices into tiles $\mathcal{T}_1, \ldots, \mathcal{T}_T$. Maintain:

- $m_i^{(t)}$: running max after tiles $1..t$.
- $\ell_i^{(t)}$: running sum of $e^{s_{ij} - m_i^{(t)}}$.
- $O_i^{(t)}$: running output.

Update rules:

$$
m_i^{(t)} = \max(m_i^{(t-1)}, \max_{j \in \mathcal{T}_t} s_{ij})
\tag{A.3}
$$

$$
\ell_i^{(t)} = \ell_i^{(t-1)} e^{m_i^{(t-1)} - m_i^{(t)}} + \sum_{j \in \mathcal{T}_t} e^{s_{ij} - m_i^{(t)}}
\tag{A.4}
$$

$$
O_i^{(t)} = O_i^{(t-1)} \cdot \frac{\ell_i^{(t-1)} e^{m_i^{(t-1)} - m_i^{(t)}}}{\ell_i^{(t)}} + \frac{\sum_{j \in \mathcal{T}_t} e^{s_{ij} - m_i^{(t)}} v_j}{\ell_i^{(t)}}
\tag{A.5}
$$

## A.3 Exactness

**Claim.** After processing all tiles, $O_i^{(T)}$ equals the exact attention 
output.

**Proof sketch.** By induction. The running update (A.5) preserves the 
invariant:

$$
O_i^{(t)} = \frac{\sum_{j \in \mathcal{T}_1 \cup \ldots \cup \mathcal{T}_t} e^{s_{ij}} v_j}{\sum_{j \in \mathcal{T}_1 \cup \ldots \cup \mathcal{T}_t} e^{s_{ij}}}
\tag{A.6}
$$

At $t = T$, this is the exact softmax attention. $\square$

## A.4 Memory Complexity

Standard attention: peak memory is $B \cdot H \cdot L^2$ for the full score 
matrix.

FlashAttention: peak memory is $B \cdot H \cdot L \cdot d_h$ for the output, 
plus tile buffers. **No full $S$ matrix.**

## A.5 FP8 Precision

E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits.

**Dynamic range:** $\sim 2^{-9}$ to $2^{8}$ with ~3% relative precision.

**Risk.** Values outside the range overflow to NaN/Inf. FA3 uses per-tile 
scaling.

## A.6 Speedup Decomposition

On Hopper, FA3 speedup comes from:

| Source | Contribution |
|---|---|
| Tiled memory | 2× (SRAM vs HBM) |
| TMA async | 1.5× (overlapped memory) |
| Warp specialization | 1.3× (compute overlap) |
| FP8 (optional) | 2× |

Combined: ~$2 \times 1.5 \times 1.3 = 3.9×$ (FP16) or ~$8×$ (FP8).

---
