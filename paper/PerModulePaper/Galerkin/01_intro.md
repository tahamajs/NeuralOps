# 1. Introduction

## 1.1 Motivation

Self-attention is often described in discrete terms:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{1.1}
$$

but its mathematical foundation is **continuous**. Attention is a 
**non-local integral operator** acting on functions over a domain $\Omega$:

$$
(\mathcal{K} Q)(s) = \int_\Omega \kappa(s, t) V(t) \, dt
\tag{1.2}
$$

where $\kappa(s, t)$ is the kernel (typically $\exp(\langle Q(s), K(t) \rangle)$). 
Standard transformers discretize this operator on the **raw token grid**, 
yielding $\mathcal{O}(N^2 d)$ complexity for $N$ tokens.

**The Galerkin idea.** In numerical analysis, integral operators are often 
discretized not on the raw grid, but on a **finite-dimensional subspace** 
spanned by a small set of **basis functions**. The **Petrov-Galerkin 
method** selects test and trial bases to minimize the residual of the 
operator equation:

$$
\langle \mathcal{K} Q - \hat{Q}, \psi_i \rangle = 0 \quad \forall i
\tag{1.3}
$$

where $\{\psi_i\}$ is the test basis.

Applying this to attention:
1. Choose an orthonormal basis $\Phi \in \mathbb{R}^{N \times r}$ (e.g., DCT).
2. Project Q, K, V into basis space: $Q_\Phi = \Phi^\top Q$, etc.
3. Compute attention in basis space at rank $r$.
4. Reconstruct via basis synthesis: $Y = \Phi \cdot O_\Phi$.

This gives $\mathcal{O}(N r d)$ complexity — **linear in $N$**.

## 1.2 The Decoder Problem

Galerkin projection is **incompatible with causal masking**. The projected 
keys $K_\Phi$ contain contributions from **all** tokens, not just the 
prefix. For an encoder this is fine; for a decoder it breaks causality.

**Solution.** For decoders, fall back to standard causal softmax attention. 
This preserves exactness at the cost of the linear-complexity benefit.

## 1.3 Problem Statement

Given a pretrained attention module with projections $W_Q, W_K, W_V, W_O$, 
we seek an augmented attention function $\mathcal{A}'$ such that:

1. **Encoder linear complexity.** For bidirectional models, 
   $\mathcal{A}'$ runs in $\mathcal{O}(N r d)$ instead of $\mathcal{O}(N^2 d)$.
2. **Decoder exactness.** For causal models, $\mathcal{A}'$ preserves the 
   standard softmax attention output exactly.
3. **Multi-architecture support.** Works on flat (DistilBERT), nested 
   (RoBERTa), causal (GPT-2), and GQA+RoPE (Qwen).
4. **Step-0 preservation.** With zero-init LoRA, initial forward matches 
   pretrained.

**Note.** Not a compression method. Parameter count increases by LoRA.

## 1.4 Our Approach

**Galerkin Attention** chooses the **discrete cosine basis**:

$$
\Phi_{n, k} = \sqrt{\frac{2 - \delta_{k0}}{N}} \cos\!\left(\frac{\pi k (2n+1)}{2N}\right)
\tag{1.4}
$$

with the DC component scaled by $1/\sqrt{N}$. The basis is orthonormalized 
via QR for numerical stability.

**Encoder forward:**

$$
\begin{aligned}
Q_\Phi &= \Phi^\top Q, \quad K_\Phi = \Phi^\top K, \quad V_\Phi = \Phi^\top V \\
\text{Scores} &= Q_\Phi K_\Phi^\top / \sqrt{d_h} \\
\text{Output}_\Phi &= \text{softmax}(\text{Scores}) V_\Phi \\
Y &= \Phi \cdot \text{Output}_\Phi \cdot \sqrt{N/r}
\end{aligned}
\tag{1.5}
$$

**Decoder forward:**

Standard causal softmax attention (no projection).

## 1.5 Contributions

- **A Galerkin formulation of self-attention.** Section 4.2.
- **A DCT-based orthonormal basis.** With energy-preserving scaling. Section 4.3.
- **Encoder-decoder dispatch.** Projection for encoders, softmax for decoders. Section 4.4.
- **Multi-architecture weight transfer.** GPT-2 Conv1D, RoBERTa nested, DistilBERT flat, Qwen GQA. Section 5.
- **Honest reporting.** We disclose that GPT-2 uses the fallback and gains no benefit. Sections 7–8.
- **Full reproducibility.** Code under {{LICENSE}}.

## 1.6 Paper Organization

Section 2 reviews related work. Section 3 provides background on integral 
operators, Galerkin methods, and DCT. Section 4 presents Galerkin Attention. 
Section 5 details architecture integration. Section 6 gives experimental 
setup. Section 7 reports results. Section 8 analyzes the encoder-decoder 
dispatch. Section 9 discusses usage. Section 10 lists limitations. Section 
11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $N$ | Sequence length |
| $r$ | Basis rank |
| $d$ | Hidden size |
| $d_h$ | Per-head dimension |
| $\Phi$ | Orthonormal basis matrix $\in \mathbb{R}^{N \times r}$ |
| $Q_\Phi$ | Projected queries $\in \mathbb{R}^{r \times d_h}$ |
| $\kappa$ | Attention kernel |
| $\Omega$ | Domain |
| $\delta_{k0}$ | Kronecker delta |
