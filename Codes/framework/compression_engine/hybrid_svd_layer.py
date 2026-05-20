"""
Hybrid SVD Layer: single‑head SVD approximation with positional encoding,
plus a full encoder layer that uses it instead of multi‑head attention.
"""

import torch
import torch.nn as nn
import numpy as np
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


class SVDSingleHead(nn.Module):
    """Single‑head SVD approximation with positional encoding."""
    def __init__(self, hidden_size, svd_rank=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.svd_rank = svd_rank
        self.register_buffer('pos_encoding', None)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape

        # Sinusoidal positional encoding (if not already present)
        if self.pos_encoding is None or self.pos_encoding.shape[0] != seq_len:
            pe = torch.zeros(seq_len, self.hidden_size, device=hidden_states.device)
            position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.hidden_size, 2).float() *
                (-np.log(10000.0) / self.hidden_size)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pos_encoding = pe.unsqueeze(0)

        x = hidden_states + self.pos_encoding[:, :seq_len, :]
        X_flat = torch.nn.functional.normalize(
            x.reshape(-1, self.hidden_size), p=2, dim=1
        )

        try:
            U, s, Vh = torch.linalg.svd(X_flat, full_matrices=False)
            k = min(self.svd_rank, len(s))
            top_s = s[:k]
            top_Vh = Vh[:k, :]
            s_max = top_s.max()
            exp_s = torch.exp(top_s - s_max)
            softmax_s = exp_s / exp_s.sum()
            weighted_U = U[:, :k] * softmax_s.view(1, -1)
            output_flat = torch.matmul(weighted_U, top_Vh)
            output = output_flat.reshape(batch_size, seq_len, self.hidden_size)
            return output
        except RuntimeError:
            return x


class DirectSVDEncoderLayer(nn.Module):
    """Encoder layer that replaces attention with SVDSingleHead + standard FFN."""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads

        self.input_layernorm = nn.LayerNorm(
            self.hidden_size, eps=getattr(config, 'layer_norm_eps', 1e-12)
        )
        self.svd_layer = SVDSingleHead(
            hidden_size=self.hidden_size,
            svd_rank=getattr(config, 'svd_rank', 20)
        )

        # FFN (same as BERT)
        self.intermediate = nn.Linear(
            self.hidden_size,
            getattr(config, 'intermediate_size', 4 * self.hidden_size)
        )
        self.output_dense = nn.Linear(
            getattr(config, 'intermediate_size', 4 * self.hidden_size),
            self.hidden_size
        )
        self.dropout = nn.Dropout(getattr(config, 'hidden_dropout_prob', 0.1))
        self.output_layernorm = nn.LayerNorm(
            self.hidden_size, eps=getattr(config, 'layer_norm_eps', 1e-12)
        )

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, output_hidden_states=False, return_dict=True):
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        svd_out = self.svd_layer(normed)
        hidden_states = residual + svd_out

        residual = hidden_states
        normed_ffn = self.output_layernorm(hidden_states)
        intermediate_out = self.intermediate(normed_ffn)
        intermediate_out = torch.nn.functional.gelu(intermediate_out)
        ff_out = self.output_dense(intermediate_out)
        ff_out = self.dropout(ff_out)
        hidden_states = residual + ff_out

        return (hidden_states,)


@COMPRESSOR_REGISTRY.register("direct_svd")
class DirectSVDCompressor(BaseCompressor):
    """
    Replaces all encoder layers with DirectSVDEncoderLayer (no attention,
    only SVD decomposition).
    """
    def __init__(self, model, svd_rank=20, **kwargs):
        super().__init__(model, name="direct_svd", **kwargs)
        self.svd_rank = svd_rank

    def compress(self):
        for name, module in self.model.named_modules():
            # We assume the model has an attribute 'encoder' with layers
            if 'encoder.layer' in name:
                # Actually we need to replace the entire layer block
                # This is simplified; a full replacement requires walking the model hierarchy
                pass
        # Because layer replacement is model‑specific, we leave a placeholder.
        # Instead we'll just demonstrate the layer exists.
        raise NotImplementedError(
            "DirectSVDCompressor needs model‑specific layer replacement. "
            "Use the DirectSVDEncoderLayer directly."
        )
    



@COMPRESSOR_REGISTRY.register("direct_svd")
def direct_svd_factory(model, comp_config):
    svd_rank = getattr(comp_config, 'direct_svd_rank', 20)
    compressor = DirectSVDCompressor(model, svd_rank=svd_rank)
    compressed_model, _ = compressor.compress()
    return compressed_model


