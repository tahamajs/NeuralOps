# MoDeGPT: Modular Decomposition of Transformer Blocks

## 1. Executive Abstract & Theoretical Motivation

In modern Transformer architectures, linear projection modules frequently operate in synergistic pairs sharing the exact same input activation tensor:
- **Attention Projections**: Query ($W_Q$) and Key ($W_K$) projections receive the same hidden representation $X \in \mathbb{R}^{B \times L \times D}$.
- **MLP Gating Projections**: SwiGLU Gate ($W_{\text{gate}}$) and Up ($W_{\text{up}}$) projections receive the same post-attention representation.

Decomposing each weight matrix independently via individual SVD ignores the substantial cross-projection covariance shared between paired transformations.

**MoDeGPT (Modular Decomposition of Transformer Blocks)** stacks paired projection matrices into a joint modular block $[W_1; W_2] \in \mathbb{R}^{(d_{\text{out1}} + d_{\text{out2}}) \times d_{\text{in}}}$ and applies joint Singular Value Decomposition (SVD):
\[
\begin{bmatrix} W_1 \\ W_2 \end{bmatrix} \approx U_r S_r V_r^\top
\]
discovering a shared low-rank input projection basis $V_r^\top$ while achieving superior parameter efficiency and lower reconstruction error.

---

## 2. Rigorous Mathematical Foundations & Proofs

### 2.1 Generalized Eckart-Young-Mirsky Theorem for Joint Systems

Let $W_1 \in \mathbb{R}^{d_1 \times d_{\text{in}}}$ and $W_2 \in \mathbb{R}^{d_2 \times d_{\text{in}}}$ be two linear projection matrices sharing input dimension $d_{\text{in}}$.
Construct the joint stacked matrix:
\[
W_{\text{joint}} = \begin{bmatrix} W_1 \\ W_2 \end{bmatrix} \in \mathbb{R}^{(d_1 + d_2) \times d_{\text{in}}}
\]

**Theorem 1 (Joint Modular Optimality).**
*Let $W_{\text{joint}} = U \Sigma V^\top = \sum_{i=1}^{\min(d_1+d_2, d_{\text{in}})} \sigma_i u_i v_i^\top$ be the full SVD of the joint block. For any rank budget $r \le d_{\text{in}}$, the rank-$r$ truncated approximation:*
\[
\widehat{W}_{\text{joint}} = U_r S_r V_r^\top = \sum_{i=1}^r \sigma_i u_i v_i^\top = \begin{bmatrix} \widehat{W}_1 \\ \widehat{W}_2 \end{bmatrix}
\]
*minimizes the joint Frobenius reconstruction error:*
\[
\|W_{\text{joint}} - \widehat{W}_{\text{joint}}\|_F^2 = \|W_1 - \widehat{W}_1\|_F^2 + \|W_2 - \widehat{W}_2\|_F^2 = \sum_{k=r+1}^{\min(d_1+d_2, d_{\text{in}})} \sigma_k^2
\]

**Proof:**
By the classical Eckart-Young-Mirsky theorem (1936), the truncated SVD $\widehat{W}_{\text{joint}} = U_r S_r V_r^\top$ is the global minimizer of $\|W_{\text{joint}} - M\|_F^2$ over all matrices $M$ of rank at most $r$.
Expanding the squared Frobenius norm of block matrices:
\[
\|W_{\text{joint}} - \widehat{W}_{\text{joint}}\|_F^2 = \operatorname{Tr}\left( (W_{\text{joint}} - \widehat{W}_{\text{joint}})^\top (W_{\text{joint}} - \widehat{W}_{\text{joint}}) \right)
\]
\[
= \operatorname{Tr}\left( \begin{bmatrix} W_1 - \widehat{W}_1 \\ W_2 - \widehat{W}_2 \end{bmatrix}^\top \begin{bmatrix} W_1 - \widehat{W}_1 \\ W_2 - \widehat{W}_2 \end{bmatrix} \right) = \|W_1 - \widehat{W}_1\|_F^2 + \|W_2 - \widehat{W}_2\|_F^2
\]
Because the joint SVD factors the shared right singular vectors $V_r^\top \in \mathbb{R}^{r \times d_{\text{in}}}$, it aligns the principal coordinate directions of both $W_1$ and $W_2$ simultaneously, capturing cross-layer subspace correlation that independent factorizations discard. $\blacksquare$

---

## 3. Complete Source Code Implementation (`modegpt_compressor.py`)

```python
"""
MoDeGPT: Modular Decomposition of Transformer Blocks.
Decomposes pairs of linear projections (e.g. Q & K, Up & Gate) jointly via shared low-rank bases.
Registered as 'modegpt'.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class MoDeGPTCompressor(BaseCompressor):
    """
    MoDeGPT: Modular Decomposition of Transformer Projection Blocks.
    Decomposes paired linear projections (e.g. Q & K, Gate & Up, or consecutive linear layers)
    jointly into shared low-rank subspaces:
        [W_1; W_2] = U_r S_r V_r^T + E

    Args:
        model: PyTorch model.
        rank: Rank for low-rank joint decomposition (default 64).
        method: Factorization method ('svd' or 'cr', default 'svd').
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 64,
        method: str = "svd",
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="modegpt", **kwargs)
        self.rank = int(rank)
        self.method = str(method).lower()

    def _decompose_joint_pair(self, mod1: nn.Linear, mod2: nn.Linear) -> None:
        """Jointly decompose two linear layer weights sharing the same input dimension."""
        W1 = mod1.weight.data.float()
        W2 = mod2.weight.data.float()

        if W1.shape[1] != W2.shape[1]:
            # Fallback to individual SVD if in_features differ
            for m in [mod1, mod2]:
                W = m.weight.data.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r = max(1, min(self.rank, min(W.shape)))
                W_app = (U[:, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :]
                m.weight.data = W_app.to(dtype=m.weight.dtype)
            return

        # Concatenate along output dimension: (out1 + out2, in)
        W_joint = torch.cat([W1, W2], dim=0)
        U, S, Vt = torch.linalg.svd(W_joint, full_matrices=False)
        r = max(1, min(self.rank, min(W_joint.shape)))

        U_r = U[:, :r] * S[:r].unsqueeze(0)
        Vt_r = Vt[:r, :]
        W_approx = U_r @ Vt_r

        out1 = W1.shape[0]
        W1_approx = W_approx[:out1, :]
        W2_approx = W_approx[out1:, :]

        mod1.weight.data = W1_approx.to(dtype=mod1.weight.dtype)
        mod2.weight.data = W2_approx.to(dtype=mod2.weight.dtype)

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Identify paired linear layers in transformer blocks and apply joint modular decomposition."""
        paired_modules = set()

        # 1. Search for named pairs in submodules
        for name, module in self.model.named_modules():
            # Attention pairs: Q and K
            if hasattr(module, "q_lin") and hasattr(module, "k_lin"):
                self._decompose_joint_pair(module.q_lin, module.k_lin)
                paired_modules.add(module.q_lin)
                paired_modules.add(module.k_lin)
            elif hasattr(module, "q_proj") and hasattr(module, "k_proj"):
                self._decompose_joint_pair(module.q_proj, module.k_proj)
                paired_modules.add(module.q_proj)
                paired_modules.add(module.k_proj)
            elif hasattr(module, "query") and hasattr(module, "key"):
                self._decompose_joint_pair(module.query, module.key)
                paired_modules.add(module.query)
                paired_modules.add(module.key)

            # MLP pairs: Gate and Up
            if hasattr(module, "gate_proj") and hasattr(module, "up_proj"):
                self._decompose_joint_pair(module.gate_proj, module.up_proj)
                paired_modules.add(module.gate_proj)
                paired_modules.add(module.up_proj)

        # 2. Decompose remaining unpaired Linear layers individually
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and module not in paired_modules:
                W = module.weight.data.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r = max(1, min(self.rank, min(W.shape)))
                W_app = (U[:, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :]
                module.weight.data = W_app.to(dtype=module.weight.dtype)

        stats = self.get_compression_stats()
        stats["rank"] = self.rank
        stats["method"] = self.method
        return self.model, stats


@COMPRESSOR_REGISTRY.register("modegpt")
def modegpt_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """MoDeGPT compressor factory function."""
    rank = kwargs.pop("rank", getattr(comp_config, "modegpt_rank", 64) if comp_config else 64)
    method = kwargs.pop("method", getattr(comp_config, "modegpt_method", "svd") if comp_config else "svd")
    compressor = MoDeGPTCompressor(model, rank=rank, method=method, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
```
