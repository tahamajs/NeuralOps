"""SVD-Mamba: SVD-factorized Selective State Space Model.
Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)
"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _unpack_weight_bias(module: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if module is None:
        return None, None
    base = getattr(module, "base_layer", module)
    w = getattr(base, "weight", None)
    b = getattr(base, "bias", None)
    if w is not None:
        w = w.data.clone()
        if (module.__class__.__name__ == "Conv1D") or (base.__class__.__name__ == "Conv1D"):
            w = w.t().contiguous()
    if b is not None:
        b = b.data.clone()
    return w, b


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos is None or sin is None:
        return q, k

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    if cos.shape[-2] != q.shape[-2]:
        cos = cos[..., : q.shape[-2], :]
        sin = sin[..., : q.shape[-2], :]

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
        k_rot_embed = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
        q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
        k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    else:
        cos_cut, sin_cut = cos[..., :head_dim], sin[..., :head_dim]
        q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
        k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))

    return q, k



class SVDLinear(nn.Module):
    is_custom_attention: bool = True
    r"""SVD-factorized linear layer with trainable LoRA adapters.
    Base weight = U @ diag(S) @ V^T.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank or max(1, min(in_features, out_features) // 4)
        self.lora_rank = lora_rank
        self.scaling = lora_alpha / max(1, lora_rank)

        self.U = nn.Parameter(torch.randn(out_features, self.rank, **factory_kwargs) * 0.02)
        self.S = nn.Parameter(torch.ones(self.rank, **factory_kwargs))
        self.V = nn.Parameter(torch.randn(in_features, self.rank, **factory_kwargs) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))
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

    def decompose_from_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        with torch.no_grad():
            W = weight.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(self.rank, len(S))
            self.U.data.zero_()
            self.S.data.zero_()
            self.V.data.zero_()
            self.U.data[:min(self.out_features, U.size(0)), :r].copy_(U[:self.out_features, :r].to(self.U.dtype))
            self.S.data[:r].copy_(S[:r].to(self.S.dtype))
            self.V.data[:min(self.in_features, Vt.size(1)), :r].copy_(Vt[:r, :self.in_features].t().contiguous().to(self.V.dtype))

            if bias is not None and self.bias is not None:
                min_b = min(self.out_features, bias.size(0))
                self.bias.data.zero_()
                self.bias.data[:min_b].copy_(bias[:min_b])

            rec = (self.U[:, :r] * self.S[:r].unsqueeze(0)) @ self.V[:, :r].t().contiguous()
            f_norm = float(torch.linalg.norm(rec))
            if f_norm < 1e-4:
                nn.init.normal_(self.U, std=0.02)
                nn.init.ones_(self.S)
                nn.init.normal_(self.V, std=0.02)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return (self.U * self.S.unsqueeze(0)) @ self.V.t().contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = ((x @ self.V) * self.S) @ self.U.t().contiguous() + self.bias
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
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        d_model: int | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 1,
        dt_rank: int | None = None,
        global_rank: int = 8,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        pretrained_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        comp_config = _cfg_get(config, "compression_config", kwargs.get("comp_config"))

        if d_model is None:
            if isinstance(config, int):
                d_model = config
            elif config is not None:
                d_model = _cfg_get(config, "hidden_size", _cfg_get(config, "d_model", _cfg_get(config, "dim", 768)))
        if d_model is None:
            d_model = kwargs.get("hidden_size", kwargs.get("d_model", 768))

        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        self.d_model = int(d_model)
        self.d_state = _cfg_get(comp_config, "d_state", d_state)
        self.d_conv = _cfg_get(comp_config, "d_conv", d_conv)
        self.expand = _cfg_get(comp_config, "expand", expand)
        self.inner_dim = self.expand * self.d_model
        self.dt_rank = dt_rank if dt_rank is not None else max(1, math.ceil(self.d_model / 16))
        self.global_rank = min(global_rank, self.d_model, self.inner_dim)
        self.lora_rank = _cfg_get(config, "lora_rank", _cfg_get(comp_config, "lora_rank", lora_rank))
        self.lora_alpha = _cfg_get(config, "lora_alpha", _cfg_get(comp_config, "lora_alpha", lora_alpha))

        # Decoder / Causal flag
        model_type = str(_cfg_get(config, "model_type", "") or "").lower()
        decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        is_dec = False
        if is_decoder is not None:
            is_dec = is_decoder
        elif config is not None:
            is_dec = bool(_cfg_get(config, "is_decoder", False))
        else:
            is_dec = bool(kwargs.get("is_causal", False))
        if any(model_type.startswith(p) for p in decoder_prefixes) or hasattr(config, "architectures") and any("GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(config, "architectures", []) or [])):
            is_dec = True
        self.is_decoder: bool = bool(is_dec)

        self.is_nested_encoder: bool = False
        num_heads = int(_cfg_get(config, "num_attention_heads", kwargs.get("num_attention_heads", 12))) if config is not None else 12
        num_kv_heads = int(_cfg_get(config, "num_key_value_heads", kwargs.get("num_key_value_heads", num_heads))) if config is not None else num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.head_dim = max(1, self.d_model // self.num_heads)
        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.supports_gqa = True

        source_attn = pretrained_attn if pretrained_attn is not None else kwargs.get("source_attn")
        target_dtype = kwargs.get("dtype")
        if target_dtype is None and source_attn is not None:
            target_dtype = next(source_attn.parameters(), torch.tensor([])).dtype
            if not isinstance(target_dtype, torch.dtype):
                target_dtype = torch.float32

        # Attention path projections with LoRA
        self.q_proj = SVDLinear(
            self.d_model, self.all_head_size,
            rank=max(1, min(self.d_model, self.all_head_size) // 4),
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True, dtype=target_dtype
        )
        self.k_proj = SVDLinear(
            self.d_model, self.kv_head_size,
            rank=max(1, min(self.d_model, self.kv_head_size) // 4),
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True, dtype=target_dtype
        )
        self.v_proj = SVDLinear(
            self.d_model, self.kv_head_size,
            rank=max(1, min(self.d_model, self.kv_head_size) // 4),
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True, dtype=target_dtype
        )

        # Gating for dynamic aggregation (negative initial bias preserves pretrained attention path)
        self.gating = nn.Linear(self.d_model, self.d_model)
        nn.init.zeros_(self.gating.weight)
        nn.init.constant_(self.gating.bias, -2.0)

        # Attention output projection
        self.attn_out_proj = SVDLinear(
            self.all_head_size, self.d_model,
            rank=max(1, min(self.all_head_size, self.d_model) // 2),
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True, dtype=target_dtype
        )

        # SVD-factorized projections
        self.in_proj = SVDLinear(
            self.d_model, 2 * self.inner_dim,
            rank=max(1, min(self.d_model, self.inner_dim) // 2),
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True, dtype=target_dtype
        )
        self.out_proj = SVDLinear(
            self.inner_dim, self.d_model,
            rank=max(1, min(self.inner_dim, self.d_model) // 2),
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True, dtype=target_dtype
        )
        self.x_proj = nn.Linear(self.inner_dim, self.dt_rank + 2 * self.d_state, bias=True)
        nn.init.uniform_(self.x_proj.weight, -0.01, 0.01)
        nn.init.zeros_(self.x_proj.bias)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -0.01, 0.01)
        nn.init.zeros_(self.dt_proj.bias)

        # Causal depthwise convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=self.d_conv,
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
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=False, dtype=target_dtype
        )

        if source_attn is not None:
            self.copy_weights(source_attn)

        if target_dtype is not None:
            self.to(target_dtype)

    @property
    def query(self) -> SVDLinear:
        return self.q_proj

    @property
    def key(self) -> SVDLinear:
        return self.k_proj

    @property
    def value(self) -> SVDLinear:
        return self.v_proj

    @property
    def q_lin(self) -> SVDLinear:
        return self.q_proj

    @property
    def k_lin(self) -> SVDLinear:
        return self.k_proj

    @property
    def v_lin(self) -> SVDLinear:
        return self.v_proj

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
                    w_o, b_o = _unpack_weight_bias(old_attn.output.dense)
            else:
                src = old_attn

            # Pattern 1: GPT-2 Fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w, c_attn_b = _unpack_weight_bias(src.c_attn)
                if c_attn_w is not None:
                    if c_attn_w.dim() == 2 and c_attn_w.shape[0] == self.d_model and c_attn_w.shape[1] == 3 * self.d_model:
                        c_attn_w = c_attn_w.t().contiguous()
                    split_size = c_attn_w.size(0) // 3
                    w_q = c_attn_w[:split_size, :]
                    w_k = c_attn_w[split_size : 2 * split_size, :]
                    w_v = c_attn_w[2 * split_size :, :]
                    if c_attn_b is not None:
                        b_q = c_attn_b[:split_size]
                        b_k = c_attn_b[split_size : 2 * split_size]
                        b_v = c_attn_b[2 * split_size :]

                c_proj_w, c_proj_b = _unpack_weight_bias(src.c_proj)
                if c_proj_w is not None:
                    if c_proj_w.dim() == 2 and c_proj_w.shape[0] != self.d_model and c_proj_w.shape[1] == self.d_model:
                        c_proj_w = c_proj_w.t().contiguous()
                    w_o = c_proj_w
                    b_o = c_proj_b

            # Pattern 2: RoBERTa / BERT separate query, key, value
            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                w_q, b_q = _unpack_weight_bias(src.query)
                w_k, b_k = _unpack_weight_bias(src.key)
                w_v, b_v = _unpack_weight_bias(src.value)

                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj"):
                        if hasattr(src, attr):
                            w_o, b_o = _unpack_weight_bias(getattr(src, attr))
                            break

            # Pattern 3: Qwen / LLaMA (q_proj, k_proj, v_proj, o_proj)
            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                w_q, b_q = _unpack_weight_bias(src.q_proj)
                w_k, b_k = _unpack_weight_bias(src.k_proj)
                w_v, b_v = _unpack_weight_bias(src.v_proj)
                if hasattr(src, "o_proj"):
                    w_o, b_o = _unpack_weight_bias(src.o_proj)

            # Pattern 4: DistilBERT (q_lin, k_lin, v_lin, out_lin)
            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                w_q, b_q = _unpack_weight_bias(src.q_lin)
                w_k, b_k = _unpack_weight_bias(src.k_lin)
                w_v, b_v = _unpack_weight_bias(src.v_lin)
                if hasattr(src, "out_lin"):
                    w_o, b_o = _unpack_weight_bias(src.out_lin)

            # Pattern 5: UniversalAdapter fallback
            if w_q is None or w_v is None:
                try:
                    adapter = UniversalAdapter()
                    w_q, b_q = adapter.get_weight_bias(src, "q")
                    w_v, b_v = adapter.get_weight_bias(src, "v")
                    if w_k is None:
                        w_k, b_k = adapter.get_weight_bias(src, "k")
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
                logger.warning(
                    f"SVDMamba: Failed to extract weights: {', '.join(missing)} from {type(old_attn)}"
                )
                return

            if w_k is None:
                w_k = w_q

            # Copy into attention path
            self.q_proj.decompose_from_weight(w_q[: self.all_head_size, : self.d_model], b_q[: self.all_head_size] if b_q is not None else None)
            self.k_proj.decompose_from_weight(w_k[: self.kv_head_size, : self.d_model], b_k[: self.kv_head_size] if b_k is not None else None)
            self.v_proj.decompose_from_weight(w_v[: self.kv_head_size, : self.d_model], b_v[: self.kv_head_size] if b_v is not None else None)

            # Assemble in_proj weight: [V, Q, V, Q] repeated to 2 * inner_dim
            if w_v.shape[0] < self.inner_dim:
                n_rep = math.ceil(self.inner_dim / w_v.shape[0])
                v_chunk = w_v.repeat(n_rep, 1)[: self.inner_dim, : self.d_model]
            else:
                v_chunk = w_v[: self.inner_dim, : self.d_model]

            if w_q.shape[0] < self.inner_dim:
                n_rep = math.ceil(self.inner_dim / w_q.shape[0])
                q_chunk = w_q.repeat(n_rep, 1)[: self.inner_dim, : self.d_model]
            else:
                q_chunk = w_q[: self.inner_dim, : self.d_model]

            in_w = torch.cat([v_chunk, q_chunk], dim=0)[: 2 * self.inner_dim, : self.d_model]

            in_b = None
            if b_v is not None and b_q is not None:
                if b_v.shape[0] < self.inner_dim:
                    n_rep = math.ceil(self.inner_dim / b_v.shape[0])
                    b_v_chunk = b_v.repeat(n_rep)[: self.inner_dim]
                else:
                    b_v_chunk = b_v[: self.inner_dim]

                if b_q.shape[0] < self.inner_dim:
                    n_rep = math.ceil(self.inner_dim / b_q.shape[0])
                    b_q_chunk = b_q.repeat(n_rep)[: self.inner_dim]
                else:
                    b_q_chunk = b_q[: self.inner_dim]

                in_b = torch.cat([b_v_chunk, b_q_chunk], dim=0)[: 2 * self.inner_dim]

            self.in_proj.decompose_from_weight(in_w, in_b)

        # Out proj (goes to attn_out_proj)
        if w_o is not None:
            out_w = w_o[: self.d_model, : self.all_head_size]
            if out_w.shape[1] < self.all_head_size:
                n_rep = math.ceil(self.all_head_size / out_w.shape[1])
                out_w = out_w.repeat(1, n_rep)[:, : self.all_head_size]
            self.attn_out_proj.decompose_from_weight(out_w, b_o[: self.d_model] if b_o is not None else None)

            # Verify Frobenius norms
            if float(torch.linalg.norm(self.in_proj.weight.float())) < 1e-4:
                logger.warning("SVDMamba: in_proj near-zero norm")
            if float(torch.linalg.norm(self.q_proj.weight.float())) < 1e-4:
                logger.warning("SVDMamba: q_proj near-zero norm")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | torch.Tensor:
        if encoder_hidden_states is not None:
            return (hidden_states, None)

        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]

        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        batch, seqlen, _ = hidden_states.shape

        mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_bool = attention_mask.bool()
            elif attention_mask.dim() == 4:
                if attention_mask.shape[-2] == attention_mask.shape[-1]:
                    mask_bool = (attention_mask[:, 0, -1, :] > -10000.0).bool()
                else:
                    mask_bool = (attention_mask[:, 0, 0, :] > -10000.0).bool()
            elif attention_mask.dim() == 3:
                mask_bool = (attention_mask[:, 0, :] > -10000.0).bool() if attention_mask.shape[1] > 1 else attention_mask.squeeze(1).bool()

        if mask_bool is None:
            mask_bool = torch.ones(batch, seqlen, device=hidden_states.device, dtype=torch.bool)

        # ---------------------------------------------------------
        # 1. Attention Path (Multi-Head Attention with RoPE and Causal Masking)
        # ---------------------------------------------------------
        Q = self.q_proj(hidden_states).view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        K = self.k_proj(hidden_states).view(batch, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        V_attn = self.v_proj(hidden_states).view(batch, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        pos_emb = position_embeddings if position_embeddings is not None else kwargs.get("position_embeddings")
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
            cos, sin = pos_emb[0], pos_emb[1]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V_attn = repeat_kv(V_attn, self.kv_repeat)

        scale = 1.0 / math.sqrt(max(1, self.head_dim))
        attn_out = None
        if mask_bool is None and not output_attentions:
            try:
                _qs = Q.half() if Q.dtype == torch.float32 else Q
                _ks = K.half() if K.dtype == torch.float32 else K
                _vs = V_attn.half() if V_attn.dtype == torch.float32 else V_attn
                _is_causal = getattr(self, 'is_decoder', False) and seqlen > 1
                _sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                    _qs, _ks, _vs, is_causal=_is_causal, dropout_p=self.dropout.p if self.training else 0.0
                )
                attn_out = _sdpa_out.to(hidden_states.dtype).transpose(1, 2).contiguous().view(batch, seqlen, self.all_head_size)
                attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e4, neginf=-1e4)
                attn_out = self.attn_out_proj(attn_out)
            except Exception:
                attn_out = None

        if attn_out is None:
            scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale

            if self.is_decoder:
                causal_mask = torch.tril(torch.ones(seqlen, seqlen, device=hidden_states.device, dtype=torch.bool))
                scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e4)

            if mask_bool is not None:
                scores = scores.masked_fill(~mask_bool.unsqueeze(1).unsqueeze(2), -1e4)

            probs = F.softmax(scores, dim=-1).to(V_attn.dtype)
            probs = torch.nan_to_num(probs, nan=0.0)
            attn_out = torch.matmul(probs, V_attn).transpose(1, 2).contiguous().view(batch, seqlen, self.all_head_size)
            attn_out = self.attn_out_proj(attn_out)

        # ---------------------------------------------------------
        # 2. Selective SSM Path (Bounded FP32 associative update with min=1.0 denominator clamp)
        # ---------------------------------------------------------
        xz = self.in_proj(hidden_states)          # [B, L, 2*inner]
        x, z = xz.chunk(2, dim=-1)                # each [B, L, inner]

        x_t = x.transpose(1, 2).contiguous()                   # [B, inner, L]
        x_pad = F.pad(x_t, (self.d_conv - 1, 0))
        x_conv = self.conv1d(x_pad)[..., :seqlen]
        x_conv = x_conv.transpose(1, 2).contiguous()           # [B, L, inner]
        x_conv = F.silu(x_conv)

        x_proj = self.x_proj(x_conv)              # [B, L, dt_rank+2*d_state]
        delta = x_proj[:, :, : self.dt_rank]
        B_ssm = x_proj[:, :, self.dt_rank : self.dt_rank + self.d_state]
        C_ssm = x_proj[:, :, self.dt_rank + self.d_state :]

        delta = self.dt_proj(delta)
        delta = F.softplus(delta + self.dt_bias).clamp(min=1e-4, max=1.0)

        A = -torch.exp(torch.clamp(self.A_log.float(), min=-20.0, max=10.0)).to(dtype=x_conv.dtype)

        log_alpha = (delta.unsqueeze(-1) * A.view(1, 1, 1, -1)).float()
        u = (delta.unsqueeze(-1) * B_ssm.unsqueeze(2) * x_conv.unsqueeze(-1)).float()
        u_den = (delta.unsqueeze(-1) * B_ssm.unsqueeze(2).abs()).float()

        chunk_size = 64
        num_chunks = (seqlen + chunk_size - 1) // chunk_size
        h_running = x_conv.new_zeros(batch, self.inner_dim, self.d_state, dtype=torch.float32)
        h_den_running = x_conv.new_zeros(batch, self.inner_dim, self.d_state, dtype=torch.float32)
        y_chunks = []

        for c in range(num_chunks):
            s_idx = c * chunk_size
            e_idx = min(seqlen, s_idx + chunk_size)

            u_c = u[:, s_idx:e_idx]
            u_den_c = u_den[:, s_idx:e_idx]
            log_a_c = log_alpha[:, s_idx:e_idx]
            lambda_c = torch.clamp(torch.cumsum(log_a_c, dim=1), min=-25.0, max=0.0)

            decay_c = torch.exp(lambda_c)
            decay_inv_c = torch.exp(-lambda_c).clamp(max=1e10)

            h_intra = decay_c * torch.cumsum(decay_inv_c * u_c, dim=1)
            h_inter = decay_c * h_running.unsqueeze(1)
            h_chunk = torch.clamp(h_intra + h_inter, min=-1e4, max=1e4)

            h_den_intra = decay_c * torch.cumsum(decay_inv_c * u_den_c, dim=1)
            h_den_inter = decay_c * h_den_running.unsqueeze(1)
            h_den_chunk = torch.clamp(h_den_intra + h_den_inter, min=0.0, max=1e4)

            # Bounded denominator clamp to prevent explosive activations
            denom_c = (C_ssm[:, s_idx:e_idx].abs().unsqueeze(2).float() * h_den_chunk).sum(dim=-1).clamp(min=1e-4)
            num_c = (C_ssm[:, s_idx:e_idx].unsqueeze(2).float() * h_chunk).sum(dim=-1)
            y_c = (num_c / denom_c).to(x_conv.dtype)
            y_chunks.append(y_c)
            h_running = h_chunk[:, -1]
            h_den_running = h_den_chunk[:, -1]

        y = torch.cat(y_chunks, dim=1)
        y = y + self.D * x_conv
        ssm_out = y * F.silu(z)

        # Global SVD mixing
        global_scores = hidden_states @ self.global_V
        global_mix = self.global_out(global_scores)
        ssm_out = ssm_out + global_mix

        ssm_proj = self.out_proj(ssm_out)

        # ---------------------------------------------------------
        # 3. Dynamic Aggregation with Negative-Biased Gating
        # ---------------------------------------------------------
        gamma = torch.sigmoid(self.gating(hidden_states))
        Y = gamma * ssm_proj + (1.0 - gamma) * attn_out
        Y = torch.nan_to_num(Y, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            out = Y
        else:
            out = Y

        if mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        if out.shape[-1] != hidden_states.shape[-1]:
            if out.shape[-1] < hidden_states.shape[-1]:
                out = F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
            else:
                out = out[..., :hidden_states.shape[-1]]

        return (out, probs if output_attentions else None)


@ATTENTION_REGISTRY.register("svd_mamba_inject")
@ATTENTION_REGISTRY.register("inject_svd_mamba")
@ATTENTION_REGISTRY.register("SVD-Mamba_inject")
@ATTENTION_REGISTRY.register("inject_SVD_Mamba")
def inject_svd_mamba(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
) -> Any | None:
    """Safe injection hook for SVDMamba with whole-model and layer support."""
    target = layer if layer is not None else model
    cfg = model_config or getattr(target, "config", None)

    # 1. Whole model traversal support
    if target is not None and (
        hasattr(target, "transformer")
        or hasattr(target, "distilbert")
        or hasattr(target, "roberta")
        or hasattr(target, "bert")
        or (hasattr(target, "model") and hasattr(target.model, "layers"))
        or (hasattr(target, "encoder") and hasattr(target.encoder, "layer"))
    ):
        layer_list = None
        if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
            layer_list = target.distilbert.transformer.layer
        elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
            layer_list = target.roberta.encoder.layer
        elif hasattr(target, "bert") and hasattr(target.bert, "encoder"):
            layer_list = target.bert.encoder.layer
        elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
            layer_list = target.transformer.h
        elif hasattr(target, "model") and hasattr(target.model, "layers"):
            layer_list = target.model.layers
        elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
            layer_list = target.encoder.layer

        if layer_list is not None:
            tot = len(layer_list)
            for idx, lyr in enumerate(layer_list):
                inject_svd_mamba(
                    layer=lyr,
                    model_config=cfg,
                    comp_config=comp_config,
                    layer_idx=idx,
                    total_layers=tot,
                    **kwargs,
                )
            return target

    # 2. Layer injection
    try:
        src = old_attn
        if src is None and target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if hasattr(cand, "query") or hasattr(cand, "q_lin") or hasattr(cand, "q_proj") or hasattr(cand, "c_attn"):
                        src = cand
                        break
            if src is None:
                for attr in ("attention", "self_attn", "attn", "self"):
                    if hasattr(target, attr):
                        src = getattr(target, attr)
                        break

        if src is None and target is not None and hasattr(target, "forward"):
            src = target

        if src is None:
            return None

        # Double-injection guard
        if src is not None:
            if isinstance(src, SVDMamba):
                return None
            if "SVDMamba" in src.__class__.__name__ or "SVD_Mamba" in src.__class__.__name__:
                return None
            if hasattr(src, "inner") and isinstance(src.inner, SVDMamba):
                return None
            if hasattr(src, "_wrapped_attention") and isinstance(src._wrapped_attention, SVDMamba):
                return None

        if target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                cand = getattr(target, attr, None)
                if cand is None:
                    continue
                if isinstance(cand, SVDMamba):
                    return None
                if "SVDMamba" in cand.__class__.__name__ or "SVD_Mamba" in cand.__class__.__name__:
                    return None
                if hasattr(cand, "inner") and isinstance(cand.inner, SVDMamba):
                    return None
                if hasattr(cand, "_wrapped_attention") and isinstance(cand._wrapped_attention, SVDMamba):
                    return None

        # Skip Qwen3.5 hybrid DeltaNet/LinearAttention layers
        if src is not None:
            c_name = src.__class__.__name__
            if "DeltaNet" in c_name or "LinearAttention" in c_name:
                return None

        d_model = (
            kwargs.get("hidden_size") or
            kwargs.get("d_model") or
            getattr(src, 'hidden_size', None) or
            getattr(src, 'embed_dim', None) or
            (src.q_proj.in_features if hasattr(src, 'q_proj') and hasattr(src.q_proj, 'in_features') else None) or
            (src.query.in_features if hasattr(src, 'query') and hasattr(src.query, 'in_features') else None) or
            (src.q_lin.in_features if hasattr(src, 'q_lin') and hasattr(src.q_lin, 'in_features') else None) or
            (target.in_features if hasattr(target, 'in_features') else None) or
            _cfg_get(cfg, 'hidden_size', None) or
            _cfg_get(cfg, 'd_model', 768)
        )

        new_module = SVDMamba(
            d_model=d_model,
            config=cfg,
            comp_config=comp_config,
            pretrained_attn=src,
            **kwargs,
        )

        if target is not None:
            if hasattr(target, "attention"):
                if hasattr(target.attention, "self"):
                    target.attention.self = new_module
                    new_module.is_nested_encoder = True
                    return new_module
                else:
                    target.attention = new_module
                    return new_module
            for attr in ("self_attn", "attn"):
                if hasattr(target, attr):
                    setattr(target, attr, new_module)
                    return new_module
            if hasattr(target, "self"):
                target.self = new_module
                new_module.is_nested_encoder = True
                return new_module

            target.attention = new_module

        return new_module
    except Exception as e:
        logger.error("inject_svd_mamba failed at layer %d: %s", layer_idx, e)
        return None


inject_svd_mamba_attention = inject_svd_mamba
inject_SVD_Mamba = inject_svd_mamba
