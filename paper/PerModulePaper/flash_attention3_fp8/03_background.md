<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Scaled Dot-Product Attention

Standard attention:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V
\tag{3.1}
$$

Memory for $QK^\top$ is $\mathcal{O}(BL^2)$ — the target of reduction.

## 3.2 Grouped-Query Attention (GQA)

GQA uses $H_{kv} < H$ to reduce KV-cache size:

$$
K, V \in \mathbb{R}^{B \times L \times H_{kv} d}, \qquad
H_{\text{repeat}} = H / H_{kv}
\tag{3.2}
$$

The `repeat_kv` operation expands KV to match Q heads.

## 3.3 FlashAttention Tiling

FlashAttention partitions $Q, K, V$ into blocks that fit in SRAM ($\sim$ 100 KB 
on A100, $\sim$ 200 KB on H100). The inner loop:

$$
O_{ij} \leftarrow \text{softmax}(Q_i K_j^\top / \sqrt{d}) V_j
\tag{3.3}
$$

accumulates partial results in SRAM, never materializing $QK^\top$ to HBM.

**Tiling cost.** For block size $S$, the number of HBM reads is 
$\mathcal{O}(BL \cdot S)$, and SRAM occupancy is $\mathcal{O}(S^2 + Sd)$.

## 3.4 FP8 Format

The 8-bit floating-point format has two variants (Micikevicius et al., 2022):

| Format | Bits (sign/exp/mant) | Max | Precision |
|---|---|---|---|
| E4M3 | 1 / 4 / 3 | 448 | ~0.5% |
| E5M2 | 1 / 5 / 2 | 57344 | ~1% |

**FP8 tensor cores.** Only Hopper (H100, H200) and Ada (RTX 4090, L40) expose 
FP8 tensor cores with 2× throughput over FP16. On Ampere (A100, RTX 30-series) 
and earlier, FP8 is only a storage format, not a compute format.

**Detection.** PyTorch exposes the FP8 dtype as `torch.float8_e4m3fn` 
(available in PyTorch ≥ 2.1). Its presence in `dir(torch)` indicates the 
build supports the type, but **does not** indicate hardware tensor-core 
support. This must be verified at runtime via CUDA capability.

## 3.5 LoRA

LoRA (Hu et al., 2021) parameterizes:

$$
\Delta W = \frac{\alpha}{r} BA, \quad
A \in \mathbb{R}^{r \times k}, \quad
B \in \mathbb{R}^{d \times r}
\tag{3.4}
$$

with $B = 0$ at init. Only $A, B$ are trained.

## 3.6 Hardware Capabilities

| GPU | Architecture | Compute Capability | FP8 Tensor Cores? |
|---|---|---|---|
| A100 | Ampere | 8.0 | ❌ |
| H100 | Hopper | 9.0 | ✅ |
| H200 | Hopper | 9.0 | ✅ |
| RTX 3090 | Ampere | 8.6 | ❌ |
| RTX 4070 | Ada | 8.9 | ⚠️ (driver-dependent) |
| RTX 4090 | Ada | 8.9 | ✅ |

On RTX 4070 Laptop (this study's hardware), FP8 tensor cores exist 
architecturally but are often not exposed through the CUDA driver; we 
detect this and fall back.

## 3.7 Notation Summary

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $L$ | Sequence length |
| $d$ | Head dimension |
| $H$ | Query head count |
| $H_{kv}$ | KV head count |
| $r$ | LoRA rank |
| $S$ | SRAM block size |
| $f_{p8}$ | FP8 format (e4m3 or e5m2) |

---