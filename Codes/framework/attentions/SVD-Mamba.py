"""SVD-Mamba: SVD-factorized Selective State Space Model.
Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.attention_mask import apply_attention_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


class SVDLinear(nn.Module):
    r"""SVD-factorized linear layer with trainable LoRA adapters.

    Base weight = U @ diag(S) @ V^T.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: Optional[int] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank or max(1, min(in_features, out_features) // 4)
        self.lora_rank = lora_rank
        self.scaling = lora_alpha / max(1, lora_rank)

        self.U = nn.Parameter(torch.randn(out_features, self.rank) * 0.02)
        self.S = nn.Parameter(torch.ones(self.rank))
        self.V = nn.Parameter(torch.randn(in_features, self.rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.U.requires_grad = False
            self.S.requires_grad = False
            self.V.requires_grad = False
            self.bias.requires_grad = False

    def decompose_from_weight(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        with torch.no_grad():
            W = weight.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(self.rank, len(S))
            self.U.data.zero_()
            self.S.data.zero_()
            self.V.data.zero_()
            self.U.data[:min(self.out_features, U.size(0)), :r].copy_(U[:self.out_features, :r].to(self.U.dtype))
            self.S.data[:r].copy_(S[:r].to(self.S.dtype))
            self.V.data[:min(self.in_features, Vt.size(1)), :r].copy_(Vt[:r, :self.in_features].t().to(self.V.dtype))

            if bias is not None and self.bias is not None:
                self.bias.data.copy_(bias[:self.out_features])

            rec = (self.U[:, :r] * self.S[:r].unsqueeze(0)) @ self.V[:, :r].t()
            f_norm = float(torch.linalg.norm(rec))
            if f_norm < 1e-4:
                raise RuntimeError(f"SVDLinear: reconstructed weight norm is near zero: {f_norm}")

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return (self.U * self.S.unsqueeze(0)) @ self.V.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = ((x @ self.V) * self.S) @ self.U.t() + self.bias
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register("svd_mamba", supports_causal=True)
@ATTENTION_REGISTRY.register("SVD-Mamba")
@ATTENTION_REGISTRY.register("svd-mamba")
@ATTENTION_REGISTRY.register("SVD_Mamba")
class SVDMamba(BaseAttention):
    r"""SVD-Mamba module with SVD-factorized projections, causal convolution, and selective SSM."""

    supports_causal: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        d_model: Optional[int] = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        global_rank: int = 8,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)

        if d_model is None:
            if isinstance(config, int):
                d_model = config
            elif config is not None:
                d_model = getattr(config, "hidden_size", getattr(config, "d_model", getattr(config, "dim", 768)))
        if d_model is None:
            d_model = kwargs.get("hidden_size", kwargs.get("d_model", 768))

        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        self.d_model = int(d_model)
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.inner_dim = expand * self.d_model
        self.dt_rank = dt_rank if dt_rank is not None else max(1, math.ceil(self.d_model / 16))
        self.global_rank = min(global_rank, self.d_model, self.inner_dim)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # Decoder / Causal flag
        if is_decoder is not None:
            self.is_decoder: bool = is_decoder
        elif config is not None:
            self.is_decoder = bool(getattr(config, "is_decoder", False))
        else:
            self.is_decoder = False

        self.is_nested_encoder: bool = False

        # SVD-factorized projections
        self.in_proj = SVDLinear(
            self.d_model, 2 * self.inner_dim,
            rank=max(1, min(self.d_model, self.inner_dim) // 2),
            lora_rank=lora_rank, lora_alpha=lora_alpha, freeze_base=True
        )
        self.out_proj = SVDLinear(
            self.inner_dim, self.d_model,
            rank=max(1, min(self.inner_dim, self.d_model) // 2),
            lora_rank=lora_rank, lora_alpha=lora_alpha, freeze_base=True
        )
        self.x_proj = SVDLinear(
            self.inner_dim, self.dt_rank + 2 * self.d_state,
            rank=self.dt_rank, lora_rank=lora_rank, lora_alpha=lora_alpha, freeze_base=True
        )
        self.dt_proj = SVDLinear(
            self.dt_rank, self.inner_dim,
            rank=self.dt_rank, lora_rank=lora_rank, lora_alpha=lora_alpha, freeze_base=True
        )

        # Causal depthwise convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=d_conv,
            groups=self.inner_dim,
            bias=False,
        )

        # State-space parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.dt_bias = nn.Parameter(torch.zeros(self.inner_dim))

        # Global SVD mixing
        self.global_V = nn.Parameter(torch.randn(self.d_model, self.global_rank) * 0.02)
        self.global_out = SVDLinear(
            self.global_rank, self.inner_dim, rank=self.global_rank,
            lora_rank=lora_rank, lora_alpha=lora_alpha, freeze_base=True
        )

    def copy_weights(self, old_attn: Any) -> None:
        """Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA."""
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            # Check nested encoder container
            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(old_attn, "query")
                and not hasattr(old_attn, "dense")
                and not hasattr(old_attn, "out_proj")
            ):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    w_o = old_attn.output.dense.weight
                    b_o = old_attn.output.dense.bias
            else:
                src = old_attn

            # Pattern 1: GPT-2 Fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w = src.c_attn.weight.detach()
                c_attn_b = src.c_attn.bias.detach() if src.c_attn.bias is not None else None
                if c_attn_w.shape[0] == self.d_model and c_attn_w.shape[1] != self.d_model:
                    c_attn_w = c_attn_w.t()

                w_q = c_attn_w[: self.d_model, :]
                w_k = c_attn_w[self.d_model : 2 * self.d_model, :]
                w_v = c_attn_w[2 * self.d_model : 3 * self.d_model, :]

                out_w = src.c_proj.weight.detach()
                if out_w.shape[0] == self.d_model and out_w.shape[1] != self.d_model:
                    out_w = out_w.t()
                w_o = out_w
                b_o = src.c_proj.bias.detach() if src.c_proj.bias is not None else None

            # Pattern 2: RoBERTa / BERT separate query, key, value
            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                w_q = src.query.weight.detach()
                b_q = src.query.bias.detach() if src.query.bias is not None else None
                w_k = src.key.weight.detach()
                b_k = src.key.bias.detach() if src.key.bias is not None else None
                w_v = src.value.weight.detach()
                b_v = src.value.bias.detach() if src.value.bias is not None else None

                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj"):
                        if hasattr(src, attr):
                            proj = getattr(src, attr)
                            w_o = proj.weight.detach()
                            b_o = proj.bias.detach() if proj.bias is not None else None
                            break

            # Pattern 3: Qwen / LLaMA (q_proj, k_proj, v_proj, o_proj)
            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                w_q = src.q_proj.weight.detach()
                b_q = src.q_proj.bias.detach() if src.q_proj.bias is not None else None
                w_k = src.k_proj.weight.detach()
                b_k = src.k_proj.bias.detach() if src.k_proj.bias is not None else None
                w_v = src.v_proj.weight.detach()
                b_v = src.v_proj.bias.detach() if src.v_proj.bias is not None else None

                if hasattr(src, "o_proj"):
                    w_o = src.o_proj.weight.detach()
                    b_o = src.o_proj.bias.detach() if src.o_proj.bias is not None else None

            # Pattern 4: DistilBERT (q_lin, k_lin, v_lin, out_lin)
            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                w_q = src.q_lin.weight.detach()
                b_q = src.q_lin.bias.detach() if src.q_lin.bias is not None else None
                w_k = src.k_lin.weight.detach()
                b_k = src.k_lin.bias.detach() if src.k_lin.bias is not None else None
                w_v = src.v_lin.weight.detach()
                b_v = src.v_lin.bias.detach() if src.v_lin.bias is not None else None

                if hasattr(src, "out_lin"):
                    w_o = src.out_lin.weight.detach()
                    b_o = src.out_lin.bias.detach() if src.out_lin.bias is not None else None

            # Pattern 5: UniversalAdapter fallback
            if w_q is None or w_v is None:
                try:
                    adapter = UniversalAdapter()
                    w_q, b_q = adapter.get_weight_bias(src, "q")
                    w_v, b_v = adapter.get_weight_bias(src, "v")
                    if w_o is None:
                        w_o, b_o = adapter.get_weight_bias(src, "o")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            # Strict validation
            missing = []
            if w_q is None:
                missing.append("Q projection")
            if w_v is None:
                missing.append("V projection")
            if w_o is None and not self.is_nested_encoder:
                missing.append("Out projection")

            if missing:
                raise RuntimeError(
                    f"SVDMamba: Failed to extract weights: {', '.join(missing)} from {type(old_attn)}"
                )

            # Assemble in_proj weight: [V, Q, V, Q] repeated to 2 * inner_dim
            chunks = [w_v, w_q, w_v, w_q]
            in_w_full = torch.cat(chunks, dim=0)
            if in_w_full.shape[0] < 2 * self.inner_dim:
                n_rep = math.ceil(2 * self.inner_dim / in_w_full.shape[0])
                in_w_full = in_w_full.repeat(n_rep, 1)
            in_w = in_w_full[: 2 * self.inner_dim, : self.d_model]

            in_b = None
            if b_v is not None and b_q is not None:
                in_b_full = torch.cat([b_v, b_q, b_v, b_q], dim=0)
                if in_b_full.shape[0] < 2 * self.inner_dim:
                    n_rep = math.ceil(2 * self.inner_dim / in_b_full.shape[0])
                    in_b_full = in_b_full.repeat(n_rep)
                in_b = in_b_full[: 2 * self.inner_dim]

            self.in_proj.decompose_from_weight(in_w, in_b)

            # Out proj
            if w_o is not None:
                out_w = w_o[: self.d_model, :]
                if out_w.shape[1] < self.inner_dim:
                    n_rep = math.ceil(self.inner_dim / out_w.shape[1])
                    out_w = out_w.repeat(1, n_rep)[:, : self.inner_dim]
                self.out_proj.decompose_from_weight(out_w, b_o[: self.d_model] if b_o is not None else None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        *args: Any,
        **kwargs: Any,
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        batch, seqlen, _ = hidden_states.shape

        # 1. Project input
        xz = self.in_proj(hidden_states)          # [B, L, 2*inner]
        x, z = xz.chunk(2, dim=-1)                # each [B, L, inner]

        # 2. Causal convolution
        x_t = x.transpose(1, 2)                   # [B, inner, L]
        x_pad = F.pad(x_t, (self.d_conv - 1, 0))
        x_conv = self.conv1d(x_pad)[..., :seqlen]
        x_conv = x_conv.transpose(1, 2)           # [B, L, inner]
        x_conv = F.silu(x_conv)

        # 3. Selective SSM (S6)
        x_proj = self.x_proj(x_conv)              # [B, L, dt_rank+2*d_state]
        delta = x_proj[:, :, : self.dt_rank]
        B = x_proj[:, :, self.dt_rank : self.dt_rank + self.d_state]
        C = x_proj[:, :, self.dt_rank + self.d_state :]

        delta = self.dt_proj(delta)
        delta = F.softplus(delta + self.dt_bias).clamp(min=1e-3, max=0.1)

        A = -torch.exp(torch.clamp(self.A_log.float(), min=-20.0, max=10.0)).to(dtype=x_conv.dtype)

        # Chunked associative scan in FP32
        log_alpha = (delta.unsqueeze(-1) * A.view(1, 1, 1, -1)).float()
        u = (delta.unsqueeze(-1) * B.unsqueeze(2) * x_conv.unsqueeze(-1)).float()

        chunk_size = 16
        num_chunks = (seqlen + chunk_size - 1) // chunk_size
        h_running = x_conv.new_zeros(batch, self.inner_dim, self.d_state, dtype=torch.float32)
        y_chunks = []

        for c in range(num_chunks):
            s_idx = c * chunk_size
            e_idx = min(seqlen, s_idx + chunk_size)

            u_c = u[:, s_idx:e_idx]
            log_a_c = log_alpha[:, s_idx:e_idx]
            lambda_c = torch.clamp(torch.cumsum(log_a_c, dim=1), min=-25.0, max=0.0)

            decay_c = torch.exp(lambda_c)
            decay_inv_c = torch.exp(-lambda_c)

            h_intra = decay_c * torch.cumsum(decay_inv_c * u_c, dim=1)
            h_inter = decay_c * h_running.unsqueeze(1)

            h_chunk = h_intra + h_inter
            y_c = (C[:, s_idx:e_idx].unsqueeze(2).float() * h_chunk).sum(dim=-1).to(x_conv.dtype)
            y_chunks.append(y_c)
            h_running = torch.clamp(h_chunk[:, -1], min=-1e4, max=1e4)

        y = torch.cat(y_chunks, dim=1)
        y = y + self.D * x_conv
        y = y * F.silu(z)

        # 4. Global SVD mixing
        global_scores = hidden_states @ self.global_V
        global_mix = self.global_out(global_scores)
        y = y + global_mix

        if self.is_nested_encoder:
            out = y
        else:
            out = self.out_proj(y)

        # Masking
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(-1).to(out.dtype)
                out = out * mask
            elif attention_mask.dim() == 4:
                mask = (attention_mask[:, 0, 0, :] > -10000.0).unsqueeze(-1).to(out.dtype)
                out = out * mask

        return (out, None)


@ATTENTION_REGISTRY.register("svd_mamba_inject")
def inject_svd_mamba(
    layer: nn.Module,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[SVDMamba]:
    """Safe injection hook for SVDMamba."""
    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    src = getattr(layer, attr)
                    break

        if src is None:
            return None

        d_model = 768
        if model_config is not None:
            d_model = getattr(model_config, "hidden_size", getattr(model_config, "dim", 768))
        elif hasattr(layer, "config"):
            d_model = getattr(layer.config, "hidden_size", 768)

        new_module = SVDMamba(d_model=d_model, config=model_config, **kwargs)
        new_module.copy_weights(src)

        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, attr):
                setattr(layer, attr, new_module)
                return new_module

        return new_module
    except Exception as e:
        logger.error("inject_svd_mamba failed at layer %d: %s", layer_idx, e)
        return None