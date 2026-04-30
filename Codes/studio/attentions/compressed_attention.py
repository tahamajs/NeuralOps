# studio/attentions/compressed_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CompressedAttentionLayer(nn.Module):
    def __init__(self, original_layer: nn.MultiheadAttention, rank_qkv: int, rank_out: int, method: str = "svd"):
        super().__init__()
        self.original_dim = original_layer.embed_dim
        self.num_heads = original_layer.num_heads
        self.head_dim = self.original_dim // self.num_heads

        w_qkv = original_layer.in_proj_weight.data.cpu().numpy()
        w_o = original_layer.out_proj.weight.data.cpu().numpy()
        U_qkv, S_qkv, Vh_qkv = np.linalg.svd(w_qkv, full_matrices=False)
        U_out, S_out, Vh_out = np.linalg.svd(w_o, full_matrices=False)
        rank_qkv = min(rank_qkv, w_qkv.shape[1])
        rank_out = min(rank_out, w_o.shape[1])

        self.U_qkv = nn.Parameter(torch.from_numpy(U_qkv[:, :rank_qkv]).float())
        self.S_qkv_sqrt = nn.Parameter(torch.from_numpy(np.sqrt(S_qkv[:rank_qkv])).float())
        self.Vh_qkv = nn.Parameter(torch.from_numpy(Vh_qkv[:rank_qkv, :]).float())
        self.U_out = nn.Parameter(torch.from_numpy(U_out[:, :rank_out]).float())
        self.S_out_sqrt = nn.Parameter(torch.from_numpy(np.sqrt(S_out[:rank_out])).float())
        self.Vh_out = nn.Parameter(torch.from_numpy(Vh_out[:rank_out, :]).float())
        self.bias_qkv = nn.Parameter(original_layer.in_proj_bias.data.clone()) if original_layer.in_proj_bias is not None else None
        self.bias_out = nn.Parameter(original_layer.out_proj.bias.data.clone()) if original_layer.out_proj.bias is not None else None
        self.dropout_p = original_layer.dropout

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True):
        x = query
        dim = self.original_dim
        projected_x = torch.matmul(x, self.Vh_qkv.t())
        qkv_compressed = torch.matmul(projected_x, self.U_qkv.t() * self.S_qkv_sqrt.view(1, -1, 1))
        q = qkv_compressed[:, :, :dim]
        k = qkv_compressed[:, :, dim:2*dim]
        v = qkv_compressed[:, :, 2*dim:]

        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn_scores += attn_mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        context = torch.matmul(attn_weights, v)

        context_fused = torch.matmul(context, self.Vh_out.t() * self.S_out_sqrt.view(1, -1, 1))
        final_output = torch.matmul(context_fused, self.U_out.t())
        if self.bias_out is not None:
            final_output += self.bias_out
        return final_output, attn_weights if need_weights else None