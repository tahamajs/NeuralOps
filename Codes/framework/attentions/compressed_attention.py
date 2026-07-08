import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("compressed_attention_layer")
class CompressedAttentionLayer(nn.Module):
    """
    SVD compression of nn.MultiheadAttention (generic).
    Compresses both the combined QKV projection and the output projection.
    """

    def __init__(
        self,
        original_layer: nn.MultiheadAttention,
        rank_qkv: int,
        rank_out: int,
        method: str = "svd",
    ):
        super().__init__()
        self.original_dim = original_layer.embed_dim
        self.num_heads = original_layer.num_heads
        self.head_dim = self.original_dim // self.num_heads
        self.method = method

        w_qkv = original_layer.in_proj_weight.data.cpu().numpy()
        w_o = original_layer.out_proj.weight.data.cpu().numpy()

        U_qkv, S_qkv, Vh_qkv = np.linalg.svd(w_qkv, full_matrices=False)
        U_out, S_out, Vh_out = np.linalg.svd(w_o, full_matrices=False)

        rank_qkv = min(rank_qkv, w_qkv.shape[1])
        rank_out = min(rank_out, w_o.shape[1])

        self.rank_qkv = rank_qkv
        self.rank_out = rank_out

        self.U_qkv = nn.Parameter(torch.from_numpy(U_qkv[:, :rank_qkv]).float())
        self.S_qkv_sqrt = nn.Parameter(
            torch.from_numpy(np.sqrt(S_qkv[:rank_qkv])).float()
        )
        self.Vh_qkv = nn.Parameter(torch.from_numpy(Vh_qkv[:rank_qkv, :]).float())

        self.U_out = nn.Parameter(torch.from_numpy(U_out[:, :rank_out]).float())
        self.S_out_sqrt = nn.Parameter(
            torch.from_numpy(np.sqrt(S_out[:rank_out])).float()
        )
        self.Vh_out = nn.Parameter(torch.from_numpy(Vh_out[:rank_out, :]).float())

        if original_layer.in_proj_bias is not None:
            self.bias_qkv = nn.Parameter(original_layer.in_proj_bias.data.clone())
        else:
            self.bias_qkv = None
        if original_layer.out_proj.bias is not None:
            self.bias_out = nn.Parameter(original_layer.out_proj.bias.data.clone())
        else:
            self.bias_out = None

        self.dropout_p = original_layer.dropout
        self.batch_first = getattr(original_layer, "batch_first", False)
        self.kdim = getattr(original_layer, "kdim", None)
        self.vdim = getattr(original_layer, "vdim", None)
        self.add_bias_kv = getattr(original_layer, "add_bias_kv", False)
        self.add_zero_attn = getattr(original_layer, "add_zero_attn", False)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
    ):
        x = query
        dim = self.original_dim

        projected_x = torch.matmul(x, self.Vh_qkv.t())
        qkv_compressed = torch.matmul(
            projected_x, self.U_qkv.t() * self.S_qkv_sqrt.view(1, -1, 1)
        )
        if self.bias_qkv is not None:
            qkv_compressed += self.bias_qkv

        q = qkv_compressed[:, :, :dim]
        k = qkv_compressed[:, :, dim : 2 * dim]
        v = qkv_compressed[:, :, 2 * dim :]

        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn_scores += attn_mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)

        context = torch.matmul(attn_weights, v)

        context_fused = torch.matmul(
            context, self.Vh_out.t() * self.S_out_sqrt.view(1, -1, 1)
        )
        final_output = torch.matmul(context_fused, self.U_out.t())
        if self.bias_out is not None:
            final_output += self.bias_out

        return final_output, attn_weights if need_weights else None


@ATTENTION_REGISTRY.register("compressed_attention_layer_energy")
class CompressedAttentionLayer2(nn.Module):
    """
    Energy‑threshold based SVD compression of nn.MultiheadAttention.
    Automatically determines rank based on cumulative singular value energy.
    """

    def __init__(
        self,
        original_layer: nn.MultiheadAttention,
        energy_threshold: float = 0.95,
        min_rank: int = 16,
    ):
        super().__init__()
        self.original_dim = original_layer.embed_dim
        self.num_heads = original_layer.num_heads
        self.head_dim = self.original_dim // self.num_heads
        self.dropout_p = original_layer.dropout
        self.batch_first = getattr(original_layer, "batch_first", False)

        w_qkv = original_layer.in_proj_weight.data.cpu().numpy()
        w_out = original_layer.out_proj.weight.data.cpu().numpy()

        U_qkv, S_qkv, Vh_qkv = np.linalg.svd(w_qkv, full_matrices=False)
        energy_qkv = np.cumsum(S_qkv) / np.sum(S_qkv)
        rank_qkv = np.argmax(energy_qkv >= energy_threshold) + 1
        rank_qkv = max(min_rank, min(rank_qkv, w_qkv.shape[1]))

        U_out, S_out, Vh_out = np.linalg.svd(w_out, full_matrices=False)
        energy_out = np.cumsum(S_out) / np.sum(S_out)
        rank_out = np.argmax(energy_out >= energy_threshold) + 1
        rank_out = max(min_rank, min(rank_out, w_out.shape[1]))

        self.rank_qkv = rank_qkv
        self.rank_out = rank_out

        self.U_qkv = nn.Parameter(torch.from_numpy(U_qkv[:, :rank_qkv]).float())
        self.S_qkv_sqrt = nn.Parameter(
            torch.from_numpy(np.sqrt(S_qkv[:rank_qkv])).float()
        )
        self.Vh_qkv = nn.Parameter(torch.from_numpy(Vh_qkv[:rank_qkv, :]).float())

        self.U_out = nn.Parameter(torch.from_numpy(U_out[:, :rank_out]).float())
        self.S_out_sqrt = nn.Parameter(
            torch.from_numpy(np.sqrt(S_out[:rank_out])).float()
        )
        self.Vh_out = nn.Parameter(torch.from_numpy(Vh_out[:rank_out, :]).float())

        if original_layer.in_proj_bias is not None:
            self.bias_qkv = nn.Parameter(original_layer.in_proj_bias.data.clone())
        else:
            self.bias_qkv = None
        if original_layer.out_proj.bias is not None:
            self.bias_out = nn.Parameter(original_layer.out_proj.bias.data.clone())
        else:
            self.bias_out = None

        self.kdim = getattr(original_layer, "kdim", None)
        self.vdim = getattr(original_layer, "vdim", None)
        self.add_bias_kv = getattr(original_layer, "add_bias_kv", False)
        self.add_zero_attn = getattr(original_layer, "add_zero_attn", False)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
    ):
        x = query
        dim = self.original_dim

        projected_x = torch.matmul(x, self.Vh_qkv.t())
        qkv_compressed = torch.matmul(
            projected_x, self.U_qkv.t() * self.S_qkv_sqrt.view(1, -1, 1)
        )
        if self.bias_qkv is not None:
            qkv_compressed += self.bias_qkv

        q = qkv_compressed[:, :, :dim]
        k = qkv_compressed[:, :, dim : 2 * dim]
        v = qkv_compressed[:, :, 2 * dim :]

        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn_scores += attn_mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)

        context = torch.matmul(attn_weights, v)

        context_fused = torch.matmul(
            context, self.Vh_out.t() * self.S_out_sqrt.view(1, -1, 1)
        )
        final_output = torch.matmul(context_fused, self.U_out.t())
        if self.bias_out is not None:
            final_output += self.bias_out

        return final_output, attn_weights if need_weights else None
