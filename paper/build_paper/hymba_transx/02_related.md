# 2. Related Work

## 2.1 Attention Alternatives

**Linear attention** (Katharopoulos et al., 2020) replaces softmax with a 
kernel feature map for $\mathcal{O}(L)$ complexity. **Performer** 
(Choromanski et al., 2021) uses orthogonal random features. Both suffer 
from approximation error on peaked distributions.

**Sparse attention** (Longformer, BigBird) reduces the attention pattern. 
**FlashAttention** (Dao et al., 2022) makes dense attention efficient on 
GPU but keeps $\mathcal{O}(L^2)$ complexity.

## 2.2 Differential Attention

**Differential Transformer** (Ye et al., 2024) introduces **differential 
attention**: two softmax attention maps are subtracted to cancel 
common-mode noise. Our implementation uses the same primitive with a 
different damping coefficient.

**Noise cancellation** is a classical signal processing technique. Ye et al. 
adapted it to attention. We combine it with SSM.

## 2.3 State Space Models

**S4** (Gu et al., 2021), **S5** (Smith et al., 2023), and **Mamba** (Gu & 
Dao, 2023) use linear time-invariant or selective SSMs for sequence 
modeling. **SimpleSSM** is a lightweight variant used in our hybrid.

**HGRN-2** (Qin et al., 2024) uses an element-wise recurrence related to 
diagonal SSMs. Our SSM branch uses similar mechanics.

## 2.4 Hybrid Attention-SSM

**Hymba** (Dong et al., 2024) combines attention and SSM heads within a 
single block, sharing key-value projections. **Jamba** (Lieber et al., 2024) 
interleaves transformer and Mamba layers. **Zamba** (Glorioso et al., 2024) 
adds shared attention blocks.

**Our distinction.** We use **differential attention** (two Q-K pairs) 
instead of standard attention, combining the benefits of both hybrid 
architecture and attention denoising.

## 2.5 Position of Our Work

| Method | Attention | SSM | Differential | Fusion |
|---|---|---|---|---|
| Standard MHSA | Single | No | No | — |
| Mamba | No | Yes | No | — |
| Hymba | Single | Yes | No | Learned |
| Diff Transformer | **Differential** | No | Yes | — |
| **HymbaTransX** | **Differential** | **Yes** | **Yes** | **Learned** |

**Key distinction.** HymbaTransX is the first block to combine **differential 
attention** with **SSM** in a single module with learned fusion.

---

## Comparison Table

| Method | Complexity | Noise cancel | Long-range | Params |
|---|---|---|---|---|
| MHSA | $O(L^2)$ | No | Yes | $4 d^2$ |
| Linear Attn | $O(L)$ | No | Yes | $4 d^2$ |
| Mamba | $O(L)$ | No | Yes | $5 d^2$ |
| Diff Transformer | $O(L^2)$ | Yes | Yes | $6 d^2$ |
| Hymba | $O(L^2 + L)$ | No | Yes | $7 d^2$ |
| **HymbaTransX** | $O(L^2 + L)$ | **Yes** | **Yes** | **$7 d^2$** |

---