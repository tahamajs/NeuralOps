# Hymba TransX: Hybrid SSM-Attention Fusion with Unified RoPE & Meta-Tokens

## 1. Executive Abstract & Architectural Motivation

Standard Transformers rely purely on Multi-Head Attention with Rotary Position Embeddings (RoPE), leading to quadratic $\mathcal{O}(L^2)$ memory growth. Hymba TransX (Hybrid Mamba-Attention Transformer Architecture) integrates parallel/interleaved SSM and MHA paths equipped with Unified RoPE (`UnifiedRoPE`) and prepended learnable meta-tokens.

By combining selective state-space linear recurrence with sparse multi-head attention and prepending learnable meta-tokens $\mathcal{M} \in \mathbb{R}^{N_{\text{meta}} \times D}$ to input sequences, Hymba TransX consolidates global memory context while retaining $\mathcal{O}(L)$ throughput. Unified RoPE provides continuous relative positional encoding across both linear SSM and attention branches.

---

## 2. Mathematical Formulations & Derivations

Given input sequence representations $X \in \mathbb{R}^{B \times L \times D}$ and learnable meta-tokens $\mathcal{M} \in \mathbb{R}^{N_{\text{meta}} \times D}$, the augmented input sequence $\widetilde{X} \in \mathbb{R}^{B \times (N_{\text{meta}} + L) \times D}$ is formed:

\[
\widetilde{X} = [\mathcal{M}; X] \in \mathbb{R}^{B \times L_{\text{aug}} \times D} \quad \text{where } L_{\text{aug}} = N_{\text{meta}} + L
\]

### 1. Unified Rotary Position Embedding (UnifiedRoPE):
For head representations $x_m \in \mathbb{R}^{d_k}$ at position $m$, UnifiedRoPE applies 2D rotation matrices to adjacent feature pairs:

\[
\Theta = \left\{ \theta_i = b^{-2(i-1)/d_k}, \, i \in [1, d_k/2] \right\}
\]
\[
R_{\Theta, m}^{d_k} x_m = \left[ x_{m, 2k} \cos(m \theta_k) - x_{m, 2k+1} \sin(m \theta_k); \; x_{m, 2k} \sin(m \theta_k) + x_{m, 2k+1} \cos(m \theta_k) \right]
\]

### 2. Dual Hybrid Path Execution:
1. **SSM Path (SimpleSSM)**:
\[
Y_{\text{ssm}} = \text{SimpleSSM}(\text{UnifiedRoPE}(\widetilde{X})) \in \mathbb{R}^{B \times L_{\text{aug}} \times D}
\]

2. **Multi-Head Attention Path (MHA)**:
\[
Q = \text{UnifiedRoPE}(\widetilde{X} W_Q), \quad K = \text{UnifiedRoPE}(\widetilde{X} W_K), \quad V = \widetilde{X} W_V
\]
\[
Y_{\text{mha}} = \text{Softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} + M_{\text{causal}} \right) V W_O \in \mathbb{R}^{B \times L_{\text{aug}} \times D}
\]

### 3. Fused Gated Output:
Outputs are combined using a learned gating parameter $\gamma \in \mathbb{R}^D$:

\[
O = \sigma(\gamma) \odot Y_{\text{ssm}} + (1 - \sigma(\gamma)) \odot Y_{\text{mha}} \in \mathbb{R}^{B \times L_{\text{aug}} \times D}
\]

Truncating meta-tokens yields final output $Y = O[:, N_{\text{meta}}:, :] \in \mathbb{R}^{B \times L \times D}$.

---

## 3. Step-by-Step Tensor Execution Trace

```
Input: hidden_states [B, L, D], Meta-Tokens [N_meta, D]
  │
  ├──> Prepend Meta-Tokens: X_aug = [Meta; hidden_states] ──> [B, N_meta + L, D]
  ├──> Apply UnifiedRoPE(X_aug) ──> [B, N_meta + L, D]
  │
  ├──> Dual Parallel Branching:
  │      ├──> Branch A (SimpleSSM): Y_ssm = SimpleSSM(X_aug) ──> [B, N_meta + L, D]
  │      └──> Branch B (MHA Attn): Y_mha = MHA(UnifiedRoPE(Q), UnifiedRoPE(K), V) ──> [B, N_meta + L, D]
  │
  ├──> Gated Fusion: O = sigmoid(gamma) * Y_ssm + (1 - sigmoid(gamma)) * Y_mha
  ├──> Truncate Meta-Tokens: Output = O[:, N_meta:, :] ──> [B, L, D]
  └──> Final Output Projection ──> [B, L, D]
```

### Detailed Tensor Transformations:

| Step | Operation | Tensor Shape | Description |
| :--- | :--- | :--- | :--- |
| 1 | Input | $[B, L, D]$ | Input sequence representation |
| 2 | Prepend Meta-Tokens | $[B, N_{\text{meta}} + L, D]$ | Concatenated sequence with meta-tokens |
| 3 | UnifiedRoPE Rotation | $[B, N_{\text{meta}} + L, D]$ | Positional 2D rotation embedding applied |
| 4 | SimpleSSM Branch | $[B, N_{\text{meta}} + L, D]$ | Linear state-space model recurrence |
| 5 | MHA Branch | $[B, N_{\text{meta}} + L, D]$ | Multi-Head Attention exact retrieval pass |
| 6 | Gated Fusion | $[B, N_{\text{meta}} + L, D]$ | Convex combination $\sigma(\gamma) Y_{\text{ssm}} + (1-\sigma(\gamma)) Y_{\text{mha}}$ |
| 7 | Meta-Token Truncation | $[B, L, D]$ | Extracted sequence output |

---

## 4. Analytical Complexity & Hardware Profiling

### FLOP and MAC Breakdown ($B$: Batch, $L$: Sequence Len, $N_{\text{meta}}$: Meta-Tokens, $D$: Dim):
- **UnifiedRoPE Embedding**: $4 B (N_{\text{meta}} + L) D$ FLOPs
- **SimpleSSM Branch**: $6 B (N_{\text{meta}} + L) D D_{\text{inner}} + 4 B (N_{\text{meta}} + L) D_{\text{inner}} d_{\text{state}}$ FLOPs
- **MHA Branch**: $8 B (N_{\text{meta}} + L) D^2 + 4 B (N_{\text{meta}} + L)^2 D$ FLOPs
- **Gated Fusion & Projection**: $2 B (N_{\text{meta}} + L) D$ FLOPs

---

## 5. Failure Modes, Rank Collapse & Edge Cases

1. **Meta-Token Over-Concentration**: Attention heads may route high probability mass exclusively to meta-tokens, starving token-to-token contextual interaction.
2. **UnifiedRoPE Frequency Mismatch**: If positional base $b$ is miscalibrated, high-frequency rotational components can induce phase noise in linear SSM updates.

---

## 6. Hardware Acceleration Blueprint & Triton Kernel Design

A Triton kernel for Hymba TransX fuses UnifiedRoPE rotation and dual-branch gating inside SRAM.

```python
# Triton Fused UnifiedRoPE & Hybrid Scan Blueprint
```

---

## 7. Next-Generation Innovations & Synthesis Opportunities

1. **Dynamic Meta-Token Allocation ($N_{\text{meta}}(x)$)**: Dynamically adjust the number of prepended meta-tokens per sequence based on context length.
2. **Hymba-Mamba MoE Hybrid**: Combine Hymba TransX dual-path execution with Expert-Choice Mixture of Experts FFN layers.
