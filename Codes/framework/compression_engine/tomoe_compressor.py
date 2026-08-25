"""
ToMoE: converts dense MLP to Mixture of Experts via dynamic structural pruning.
Registered as 'tomoe'.
"""
from typing import Any

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from sklearn.cluster import KMeans


class ToMoELinear(nn.Module):
    """A single MoE‑fied linear layer: gating + experts."""

    def __init__(self, original: nn.Linear, num_experts: int, top_k: int):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.num_experts = num_experts
        self.top_k = top_k

        # Cluster output neurons based on weight similarity
        W = original.weight.data.float().cpu().numpy()
        kmeans = KMeans(n_clusters=num_experts, random_state=42, n_init="auto")
        assignments = kmeans.fit_predict(W)  # (out_features,)

        # Create expert masks and biases
        self.experts = nn.ModuleList()
        self.masks = []
        for e in range(num_experts):
            mask = torch.tensor(assignments == e)  # (out_features,)
            self.masks.append(mask)
            expert = nn.Linear(
                self.in_features, mask.sum().item(), bias=original.bias is not None
            )
            # Copy the relevant rows of the original weight
            expert.weight.data = original.weight.data[mask].clone()
            if original.bias is not None:
                expert.bias.data = original.bias.data[mask].clone()
            self.experts.append(expert)

        # Router (learned)
        self.router = nn.Linear(self.in_features, num_experts)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        device = self.experts[0].weight.device
        dtype = self.experts[0].weight.dtype
        W = torch.zeros(self.out_features, self.in_features, device=device, dtype=dtype)
        for e, expert in enumerate(self.experts):
            mask = self.masks[e].to(device)
            W[mask] = expert.weight
        return W

    def forward(self, x):
        # x: (..., in_features)
        """Forward pass."""
        logits = self.router(x)  # (..., E)
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        topk_vals = torch.softmax(topk_vals, dim=-1)

        out = torch.zeros(
            *x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype
        )
        for k in range(self.top_k):
            expert_idx = topk_idx[..., k]  # (...)
            scale = topk_vals[..., k : k + 1]  # (..., 1)
            for e in range(self.num_experts):
                mask = expert_idx == e
                if mask.any():
                    expert_out = self.experts[e](x)  # (..., out_e)
                    # Scatter to full dimension
                    full_out = torch.zeros(
                        *x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype
                    )
                    full_out[..., self.masks[e]] = expert_out
                    out = out + mask.unsqueeze(-1).float() * scale * full_out
        return out


class ToMoECompressor(BaseCompressor):
    """
    Converts FFN layers into Mixture of Experts via weight clustering.

    Args:
        model: PyTorch model.
        num_experts: number of experts per MoE layer (default 4).
        top_k: number of active experts per token (default 2).
    """

    def __init__(self, model, num_experts=4, top_k=2, **kwargs):
        super().__init__(model, name="tomoe", **kwargs)
        self.num_experts = num_experts
        self.top_k = top_k

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and (
                "ffn" in name.lower() or "intermediate" in name.lower()
            ):
                parent = self._get_parent(name)
                if parent is not None:
                    child_name = name.split(".")[-1]
                    moe_layer = ToMoELinear(module, self.num_experts, self.top_k)
                    setattr(parent, child_name, moe_layer)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()

    def _get_parent(self, full_name) -> Any:
        """ get parent."""
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None
        return parent
