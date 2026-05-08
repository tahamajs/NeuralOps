import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY

class CompressedDistilBertAttention(nn.Module):
    def __init__(self, model_config, rank_qkv=32, rank_out=32, dropout=0.1, pretrained_attn=None):
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required")
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.hidden_size = model_config.hidden_size
        self.dropout = dropout

        w_q = pretrained_attn.q_lin.weight.data.cpu().numpy()
        w_k = pretrained_attn.k_lin.weight.data.cpu().numpy()
        w_v = pretrained_attn.v_lin.weight.data.cpu().numpy()
        w_qkv = np.concatenate([w_q, w_k, w_v], axis=0)
        w_out = pretrained_attn.out_lin.weight.data.cpu().numpy()

        rank_qkv = max(1, min(rank_qkv, w_qkv.shape[0], w_qkv.shape[1]))
        rank_out = max(1, min(rank_out, w_out.shape[0], w_out.shape[1]))

        U_qkv, S_qkv, Vh_qkv = np.linalg.svd(w_qkv, full_matrices=False)
        U_out, S_out, Vh_out = np.linalg.svd(w_out, full_matrices=False)

        self.register_buffer('U_qkv', torch.from_numpy(U_qkv[:, :rank_qkv]).float())
        self.register_buffer('S_qkv', torch.from_numpy(S_qkv[:rank_qkv]).float())
        self.register_buffer('Vh_qkv', torch.from_numpy(Vh_qkv[:rank_qkv, :]).float())
        self.register_buffer('U_out', torch.from_numpy(U_out[:, :rank_out]).float())
        self.register_buffer('S_out', torch.from_numpy(S_out[:rank_out]).float())
        self.register_buffer('Vh_out', torch.from_numpy(Vh_out[:rank_out, :]).float())

        self.bias_q = pretrained_attn.q_lin.bias.data.clone() if pretrained_attn.q_lin.bias is not None else None
        self.bias_k = pretrained_attn.k_lin.bias.data.clone() if pretrained_attn.k_lin.bias is not None else None
        self.bias_v = pretrained_attn.v_lin.bias.data.clone() if pretrained_attn.v_lin.bias is not None else None
        self.bias_out = pretrained_attn.out_lin.bias.data.clone() if pretrained_attn.out_lin.bias is not None else None

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        device = hidden_states.device
        if self.U_qkv.device != device:
            self.U_qkv = self.U_qkv.to(device)
            self.S_qkv = self.S_qkv.to(device)
            self.Vh_qkv = self.Vh_qkv.to(device)
            self.U_out = self.U_out.to(device)
            self.S_out = self.S_out.to(device)
            self.Vh_out = self.Vh_out.to(device)
            if self.bias_q is not None:
                self.bias_q = self.bias_q.to(device)
                self.bias_k = self.bias_k.to(device)
                self.bias_v = self.bias_v.to(device)
                self.bias_out = self.bias_out.to(device)

        B, N, _ = hidden_states.shape
        proj = torch.matmul(hidden_states, self.Vh_qkv.t())
        qkv_comp = torch.matmul(proj, self.U_qkv.t() * self.S_qkv.view(1, -1, 1))
        dim = self.hidden_size          # <-- ADD THIS LINE
        q = qkv_comp[:, :, :dim]
        k = qkv_comp[:, :, dim:2*dim]
        v = qkv_comp[:, :, 2*dim:]

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2,-1)) * scale
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
            else:
                mask = attention_mask
            scores = scores + (1.0 - mask) * -1e9

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        context = torch.matmul(attn, v).transpose(1,2).contiguous().view(B, N, dim)

        fused = torch.matmul(context, self.Vh_out.t() * self.S_out.view(1, -1, 1))
        out = torch.matmul(fused, self.U_out.t())
        if self.bias_out is not None:
            out = out + self.bias_out
        return out, None

@ATTENTION_REGISTRY.register("compressed_mha")
def inject_compressed_mha(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    rank_qkv = getattr(comp_config, 'svd_rank_qkv', 32)
    rank_out = getattr(comp_config, 'svd_rank_out', 32)
    dropout = getattr(comp_config, 'dropout', 0.1)
    new_attn = CompressedDistilBertAttention(model_config, rank_qkv, rank_out, dropout, old_attn)
    layer.attention = new_attn
    return new_attn