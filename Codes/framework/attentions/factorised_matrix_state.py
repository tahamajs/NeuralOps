import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("factorised_matrix_state")
class FactorisedMatrixState(nn.Module):
    def __init__(self, config, rank=32):
        super().__init__()
        self.d_model = config.hidden_size
        self.rank = rank if rank is not None else 32
        self.L = nn.Parameter(torch.randn(self.d_model, self.rank) * 0.01)
        self.R = nn.Parameter(torch.randn(self.d_model, self.rank) * 0.01)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # This is a stateful module; for simplicity, we ignore k,v and use hidden_states as query
        B, N, D = hidden_states.shape
        # We'll process each token sequentially (simplified)
        self.L @ self.R.T  # (D, D)
        outputs = []
        state = torch.zeros(B, D, D, device=hidden_states.device)
        for t in range(N):
            xt = hidden_states[:, t, :]  # (B, D)
            state = state + (xt.unsqueeze(-1) @ xt.unsqueeze(1))
            out_t = (state @ xt.unsqueeze(-1)).squeeze(-1)
            outputs.append(out_t)
        out = torch.stack(outputs, dim=1)  # (B, N, D)
        return out, None
