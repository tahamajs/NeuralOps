# framework/attentions/ssm/gated_delta_net.py
"""
Recurrent Gated DeltaNet layer with learnable forget gate.
Based on ICLR 2025 Gated Delta Networks (simplified recurrent version).
"""

import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


@ATTENTION_REGISTRY.register("gated_delta_net")
class GatedDeltaNetLayer(nn.Module):
    def __init__(self, d_model: int, use_spectral_norm: bool = False):
        super().__init__()
        self.forget_gate = nn.Linear(d_model, d_model)
        self.input_gate = nn.Linear(d_model, d_model)
        self.hidden_transform = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        if use_spectral_norm:
            nn.utils.spectral_norm(self.forget_gate)
            nn.utils.spectral_norm(self.input_gate)
            nn.utils.spectral_norm(self.hidden_transform)
            nn.utils.spectral_norm(self.out_proj)

    def forward(self, x: torch.Tensor, state: torch.Tensor = None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        if state is None:
            state = torch.zeros(batch, d_model, device=x.device)

        outputs = []
        for t in range(seq_len):
            xt = x[:, t]
            f = torch.sigmoid(self.forget_gate(xt))
            i = torch.sigmoid(self.input_gate(xt))
            h_cand = torch.tanh(self.hidden_transform(xt))
            state = f * state + i * h_cand
            outputs.append(self.out_proj(state))

        return torch.stack(outputs, dim=1)