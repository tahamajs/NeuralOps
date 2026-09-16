# 2. Related Work

## 2.1 XAttention

**XAttention** (Xu et al., 2025) proposes **training-free** block scoring via 
antidiagonal sub-sampling. Blocks are ranked by the mean magnitude of their 
antidiagonal entries. We adopt the sub-sampled block scoring approach.

## 2.2 SeerAttention

**SeerAttention** (Gao et al., 2025) introduces a **learnable** gating network 
for block-level sparsity. It predicts per-head sparsity budgets using a 
low-rank query-key pooling network. We adopt a similar gating structure.

## 2.3 DuSA

**DuSA** (Cao et al., 2025) uses a **dual-stage** approach: dense intra-block 
attention (all tokens within a block) + sparse inter-block attention (top-$k$ 
blocks). We adopt this two-stage structure.

## 2.4 AnchorAttention

**AnchorAttention** (Zhu et al., 2025) applies **stripe band masking** at the 
token level. Tokens within a stripe group attend to adjacent groups. We adopt 
a simplified stripe mask.

## 2.5 Spectraformer

**Spectraformer** (Nguyen et al., 2025) uses RFF for global linear attention. 
We apply it periodically (every `rectify_interval` layers) as a global 
rectification.

## 2.6 Softpick

**Softpick** (Zuhri et al., 2025) is a rectified softmax that produces 
natural sparsity (~47% activation) without sink tokens. We use it in place 
of standard softmax.

## 2.7 Position of Our Work

| Method | Learnable? | Training-free? | Complexity |
|---|---|---|---|
| XAttention | No | Yes | $\mathcal{O}(N B_s + N k B_s)$ |
| SeerAttention | Yes | No | $\mathcal{O}(N B_s + N k B_s)$ |
| DuSA | Partial | No | $\mathcal{O}(N B_s + N k B_s)$ |
| AnchorAttention | No | Yes | $\mathcal{O}(N S_w)$ |
| Spectraformer | No | Yes | $\mathcal{O}(N F)$ |
| **NEXUS** | **Yes** | **Partial** | **$\mathcal{O}(N B_s + N k B_s)$** |

**Key distinction.** NEXUS is the first to combine all six in a single 
module with modular enable/disable flags.

---