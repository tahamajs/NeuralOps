"""
Shared Orthogonal Attention (SOA) with Per-Projection Latent Bottlenecks.
Fixes:
- Consistent mask handling (bool/0-1/additive).
- Copy bias for out_proj.
- Support output_attentions.
"""
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("soa_module")
class SOAAttention(nn.Module):
    def __init__(
        self,
        model_config: Any,
        k_rank: int,
        dropout: float = 0.1,
        pretrained_attn: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero-shot SVD initialization.")

        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.hidden_size = model_config.hidden_size
        self.k_rank = k_rank

        self.q_down = nn.Linear(self.hidden_size, k_rank, bias=False)
        self.k_down = nn.Linear(self.hidden_size, k_rank, bias=False)
        self.v_down = nn.Linear(self.hidden_size, k_rank, bias=False)

        self.q_up = nn.Linear(k_rank, self.hidden_size, bias=False)
        self.k_up = nn.Linear(k_rank, self.hidden_size, bias=False)
        self.v_up = nn.Linear(k_rank, self.hidden_size, bias=False)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)

        self._init_from_pretrained(pretrained_attn)

    def _init_from_pretrained(self, pretrained_attn: Any) -> None:
        with torch.no_grad():
            for orig, down, up in [
                (pretrained_attn.q_lin, self.q_down, self.q_up),
                (pretrained_attn.k_lin, self.k_down, self.k_up),
                (pretrained_attn.v_lin, self.v_down, self.v_up),
            ]:
                W = orig.weight.data.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r = min(self.k_rank, U.size(1), Vt.size(0))
                down.weight.data[:r, :] = Vt[:r, :].to(down.weight.dtype)
                up.weight.data[:, :r] = (U[:, :r] * S[:r]).to(up.weight.dtype)

            self.out_proj.weight.data.copy_(pretrained_attn.out_lin.weight)
            if pretrained_attn.out_lin.bias is not None:
                self.out_proj.bias.data.copy_(pretrained_attn.out_lin.bias)

    def _convert_mask_to_additive(self, mask, B, N, device, dtype):
        """Return additive mask [B,1,1,N] with 0 for valid, -10000 for pad."""
        if mask is None:
            return None
        if mask.dim() == 2:
            if mask.dtype == torch.bool:
                valid = mask
            else:
                valid = mask > 0
            return (~valid).unsqueeze(1).unsqueeze(2).to(dtype) * -10000.0
        elif mask.dim() == 4:
            if mask.dtype == torch.bool:
                valid = mask[:, 0, 0, :]
                return (~valid).unsqueeze(1).unsqueeze(2).to(dtype) * -10000.0
            else:
                # Assume already additive (0 and -inf/-10000)
                return mask.to(dtype)
        else:
            raise ValueError(f"Unsupported attention_mask shape: {mask.shape}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape

        z_q = self.q_down(hidden_states)
        z_k = self.k_down(hidden_states)
        z_v = self.v_down(hidden_states)

        q = self.q_up(z_q)
        k = self.k_up(z_k)
        v = self.v_up(z_v)

        H = self.num_heads
        D = self.head_dim
        q = q.view(B, N, H, D).transpose(1, 2)
        k = k.view(B, N, H, D).transpose(1, 2)
        v = v.view(B, N, H, D).transpose(1, 2)

        scale = D ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        additive_mask = self._convert_mask_to_additive(attention_mask, B, N, scores.device, scores.dtype)
        if additive_mask is not None:
            scores = scores + additive_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, N, H * D)

        output = self.out_proj(context)

        return output, attn_weights if output_attentions else None


@ATTENTION_REGISTRY.register("soa_attn")
def inject_soa_attn(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    k_rank = getattr(comp_config, "soa_rank", 32)
    dropout = getattr(comp_config, "dropout", 0.1)
    new_attn = SOAAttention(model_config, k_rank, dropout, old_attn)
    layer.attention = new_attn
    return new_attn