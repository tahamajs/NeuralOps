"""
Linformer Attention – corrected version.
Fixes:
- Use pretrained_attn for initializing Q,K,V,out_proj.
- Add injection function.
- Handle longer sequences by padding E,F (though limited).
- Proper mask handling.
"""
import math
from typing import Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.attention_mask import get_token_valid_mask
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("linformer")
class LinformerAttention(nn.Module):
    def __init__(
        self,
        config: Any,
        k: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim

        self.k_proj_dim = getattr(config, "linformer_k", k if k is not None else getattr(config, "k", 64))
        self.max_seq_len = getattr(config, "max_position_embeddings", max_seq_len if max_seq_len is not None else 512)

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)

        # Initialize from pretrained if available
        if pretrained_attn is not None:
            with torch.no_grad():
                self.query.weight.data.copy_(pretrained_attn.q_lin.weight.data)
                self.query.bias.data.copy_(pretrained_attn.q_lin.bias.data)
                self.key.weight.data.copy_(pretrained_attn.k_lin.weight.data)
                self.key.bias.data.copy_(pretrained_attn.k_lin.bias.data)
                self.value.weight.data.copy_(pretrained_attn.v_lin.weight.data)
                self.value.bias.data.copy_(pretrained_attn.v_lin.bias.data)
                self.out_proj.weight.data.copy_(pretrained_attn.out_lin.weight.data)
                self.out_proj.bias.data.copy_(pretrained_attn.out_lin.bias.data)

        self.E = nn.Parameter(torch.empty(self.k_proj_dim, self.max_seq_len))
        self.F = nn.Parameter(torch.empty(self.k_proj_dim, self.max_seq_len))
        nn.init.normal_(self.E, std=1.0 / math.sqrt(self.k_proj_dim))
        nn.init.normal_(self.F, std=1.0 / math.sqrt(self.k_proj_dim))

        dropout_p = getattr(config, "dropout", getattr(config, "hidden_dropout_prob", getattr(config, "attention_probs_dropout_prob", 0.1)))
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2)
        K = self.key(hidden_states).view(B, N, H, d).transpose(1, 2)
        V = self.value(hidden_states).view(B, N, H, d).transpose(1, 2)

        # Mask padding (valid_mask from framework)
        valid_mask = get_token_valid_mask(attention_mask, B, N, device)  # [B,N] bool True for valid
        mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)
        K = K * mask_expanded
        V = V * mask_expanded

        # Slice or pad E,F
        if N <= self.max_seq_len:
            E_act = self.E[:, :N]
            F_act = self.F[:, :N]
        else:
            # If longer, pad E,F with zeros (limitation; better to increase max_seq_len)
            E_act = F.pad(self.E, (0, N - self.max_seq_len))
            F_act = F.pad(self.F, (0, N - self.max_seq_len))

        # Project sequence dimension: (k, N) @ (B,H,N,d) -> (B,H,k,d)
        K_proj = torch.matmul(E_act, K)  # E_act: [k,N], K: [B,H,N,d]
        V_proj = torch.matmul(F_act, V)

        scale = d ** -0.5
        scores = torch.matmul(Q, K_proj.transpose(-2, -1)) * scale  # [B,H,N,k]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        if head_mask is not None:
            attn = attn * head_mask

        context = torch.matmul(attn, V_proj)  # [B,H,N,d]
        context = context * mask_expanded  # mask query outputs
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        out = self.out_proj(context)

        return out, attn if output_attentions else None


@ATTENTION_REGISTRY.register("linformer_inject")
def inject_linformer(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    k = getattr(comp_config, "linformer_k", 64)
    max_seq_len = getattr(model_config, "max_position_embeddings", 512)
    new_attn = LinformerAttention(model_config, k=k, max_seq_len=max_seq_len, pretrained_attn=old_attn)
    layer.attention = new_attn
    return new_attn