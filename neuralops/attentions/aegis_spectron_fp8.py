"""
Aegis-Spectron-FP8 Hybrid Attention .

Three-branch attention with spectral entropy routing:
  1. Local sliding-window attention  (O(N*W))
  2. Global low-rank ELU+1 linear attention (O(N*r*D))
  3. Linear state recurrence (O(N*D), chunked, numerically stable)

"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from neuralops.core.attention_utils import repeat_kv
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

# ===========================================================================
# FP8 simulation (SP-FIX-01, SP-FIX-14)
# ===========================================================================


def fp8_cast_simulate(
    x: torch.Tensor,
    format: str = "e4m3",
    per_channel: bool = False,
    use_ste: bool | None = None,
) -> torch.Tensor:
    """
    Simulate FP8 quantization error via dynamic scaling + cast-dequantize roundtrip.

    Args:
        x: input tensor (any floating dtype).
        format: "e4m3" (max 448) or "e5m2" (max 57344).
        per_channel: per-last-dim scaling if True, else per-tensor.
        use_ste: If True, forward returns quantized value, gradient flows through
                 the identity (straight-through estimator). If False, forward
                 returns quantized value with no STE. If None, auto: STE in
                 training, plain quantize in eval.
    """
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        return x

    if use_ste is None:
        use_ste = x.requires_grad and torch.is_grad_enabled()

    orig_dtype = x.dtype
    x_f = x.float()

    fmt = format.lower()
    if fmt in ("e4m3", "float8_e4m3fn"):
        max_val = 448.0
        torch_dtype = getattr(torch, "float8_e4m3fn", None)
        mantissa_bits = 3
    elif fmt in ("e5m2", "float8_e5m2"):
        max_val = 57344.0
        torch_dtype = getattr(torch, "float8_e5m2", None)
        mantissa_bits = 2
    else:
        raise ValueError(f"Unsupported FP8 format: {format!r}")

    amax = (x_f.abs().amax(dim=-1, keepdim=True) if per_channel else x_f.abs().amax()).clamp(
        min=1e-12
    )
    scale = max_val / amax
    x_scaled = (x_f * scale).clamp(min=-max_val, max=max_val)

    x_dequant: torch.Tensor | None = None

    if torch_dtype is not None:
        try:
            x_fp8 = x_scaled.to(torch_dtype)
            x_dequant = (x_fp8.float() / scale.float().clamp(min=1e-8)).to(orig_dtype)
        except Exception:
            x_dequant = None

    if x_dequant is None:
        abs_x = x_scaled.abs().clamp(min=1e-12)
        exp = torch.floor(torch.log2(abs_x))
        factor = 2.0 ** (exp - mantissa_bits)
        rounded = torch.round(abs_x / factor) * factor
        x_dequant = (
            (torch.sign(x_scaled) * rounded).clamp(min=-max_val, max=max_val)
            / scale.float().clamp(min=1e-8)
        ).to(orig_dtype)

    if use_ste:
        # Forward = quantized value; gradient = identity.
        return x + (x_dequant - x).detach()
    return x_dequant


# ===========================================================================
# Decoder detection (SP-FIX-07)
# ===========================================================================


def _detect_decoder(
    config: Any,
    is_decoder: bool | None,
    old_attn: nn.Module | None,
) -> bool:
    if is_decoder is not None:
        return bool(is_decoder)
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "causal", False):
            return True
        mt = str(getattr(config, "model_type", "")).lower()
        if mt.startswith(
            (
                "gpt",
                "qwen",
                "llama",
                "mistral",
                "gemma",
                "phi",
                "opt",
                "bloom",
                "falcon",
                "olmo",
                "mpt",
                "stablelm",
            )
        ):
            return True
    if old_attn is not None:
        if getattr(old_attn, "is_decoder", False):
            return True
        if hasattr(old_attn, "c_attn"):
            return True
        if "gpt" in old_attn.__class__.__name__.lower():
            return True
    return False


# ===========================================================================
# Config (SP-FIX-08)
# ===========================================================================


class AegisSpectronFP8Config:
    def __init__(self, **kwargs: Any):
        self.hidden_size = int(
            kwargs.get("hidden_size", kwargs.get("d_model", kwargs.get("dim", 768)))
        )
        self.num_heads = int(
            kwargs.get("num_heads", kwargs.get("num_attention_heads", kwargs.get("n_heads", 4)))
        )
        self.num_key_value_heads = int(
            kwargs.get("num_key_value_heads", kwargs.get("num_kv_heads", self.num_heads))
        )
        self.head_dim = int(
            kwargs.get(
                "head_dim",
                kwargs.get("kv_channels", self.hidden_size // max(1, self.num_heads)),
            )
        )
        self.window_size = int(
            kwargs.get("window_size", kwargs.get("local_window", kwargs.get("block_size", 64)))
        )
        self.svd_rank = int(kwargs.get("svd_rank", kwargs.pop("rank", min(128, self.hidden_size))))
        self.lora_rank = int(kwargs.get("lora_rank", kwargs.get("lora_r", 32)))
        self.temperature = float(kwargs.get("temperature", 1.0))
        self.lambda_cost = float(kwargs.get("lambda_cost", 0.1))
        self.c_loc = float(kwargs.get("c_loc", 0.5))
        self.c_glob = float(kwargs.get("c_glob", 0.8))
        self.c_lin = float(kwargs.get("c_lin", 1.0))
        self.dropout = float(kwargs.get("dropout", 0.1))
        self.router_init_bias = tuple(kwargs.get("router_init_bias", (4.0, 0.0, 0.0)))
        self.is_decoder = bool(kwargs.get("is_decoder", False))
        self.fp8_format = str(kwargs.get("fp8_format", "e4m3"))
        self.chunk_size = int(kwargs.get("chunk_size", 64))


# ===========================================================================
# FactorizedLinear
# ===========================================================================


class FactorizedLinear(nn.Module):
    is_custom_attention = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        lora_rank: int = 32,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(1, min(int(rank), self.in_features, self.out_features))
        self.svd_rank = self.rank
        self.lora_rank = (
            min(int(lora_rank), min(self.in_features, self.out_features)) if lora_rank > 0 else 0
        )

        factory_kwargs = {"device": device, "dtype": dtype}
        self.lin1 = nn.Linear(self.in_features, self.rank, bias=False, **factory_kwargs)
        self.lin2 = nn.Linear(self.rank, self.out_features, bias=bias, **factory_kwargs)

        self.lin1.weight.requires_grad = False
        self.lin2.weight.requires_grad = False
        if self.lin2.bias is not None:
            self.lin2.bias.requires_grad = False

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty(self.lora_rank, self.in_features, **factory_kwargs)
            )
            self.lora_B = nn.Parameter(
                torch.empty(self.out_features, self.lora_rank, **factory_kwargs)
            )
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lin1.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lin2.weight, a=math.sqrt(5))
            if self.lin2.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.lin2.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.lin2.bias, -bound, bound)
            if self.lora_A is not None:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)

    @property
    def weight(self) -> torch.Tensor:
        w = torch.matmul(self.lin2.weight, self.lin1.weight)
        if self.lora_A is not None:
            w = w + torch.matmul(self.lora_B, self.lora_A)
        return w

    @property
    def bias(self) -> torch.Tensor | None:
        return self.lin2.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_down = x.to(self.lin1.weight.dtype)
        latent = self.lin1(x_down)
        latent_up = latent.to(self.lin2.weight.dtype)
        out = self.lin2(latent_up)
        if self.lora_A is not None:
            a = self.lora_A.to(orig_dtype) if self.lora_A.dtype != orig_dtype else self.lora_A
            b = self.lora_B.to(orig_dtype) if self.lora_B.dtype != orig_dtype else self.lora_B
            out = out.to(orig_dtype) + F.linear(F.linear(x, a), b)
        return out.to(orig_dtype)


# ===========================================================================
# Router (SP-FIX-12, SP-FIX-16)
# ===========================================================================


class SpectronSpectralEntropyRouter(nn.Module):
    def __init__(self, config: AegisSpectronFP8Config):
        super().__init__()
        self.tau = float(config.temperature)
        self.lam = float(config.lambda_cost)
        self.register_buffer("costs", torch.tensor([config.c_loc, config.c_glob, config.c_lin]))
        self.h_proj = nn.Linear(config.hidden_size, 3)
        nn.init.zeros_(self.h_proj.weight)
        self.h_proj.bias.data.copy_(
            torch.tensor(config.router_init_bias, dtype=self.h_proj.bias.dtype)
        )

    def forward(self, X: torch.Tensor, token_valid: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = X.shape

        with torch.no_grad():
            S_sq = X.float().pow(2)
            denom = S_sq.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            p = (S_sq / denom).clamp(min=1e-8, max=1.0)
            entropy = -(p * torch.log(p)).sum(dim=-1)  # (B, N)

            if token_valid is not None:
                entropy = entropy * token_valid.float()
                entropy = entropy / entropy.amax(dim=-1, keepdim=True).clamp(min=1e-6)

        h = self.h_proj(X.to(self.h_proj.weight.dtype))  # (B, N, 3)
        h = h * entropy.unsqueeze(-1).to(h.dtype)

        c = self.costs.to(h.dtype)
        logits = (h - self.lam * c) / max(self.tau, 1e-4)
        return F.softmax(logits.float(), dim=-1).to(X.dtype)


# ===========================================================================
# Local sliding-window attention (SP-FIX-10)
# ===========================================================================


def _local_window_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window: int,
    token_valid: torch.Tensor | None,
    is_decoder: bool,
    past_len: int,
    dropout_p: float,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, h, N, d = Q.shape
    N_k = K.shape[2]
    radius = window // 2
    scale = d**-0.5

    abs_q = (past_len + torch.arange(N, device=Q.device)).unsqueeze(1)  # (N, 1)
    abs_k = torch.arange(N_k, device=Q.device).unsqueeze(0)  # (1, N_k)

    band = (abs_q - abs_k).abs() <= radius
    band = band.unsqueeze(0).unsqueeze(0)

    if is_decoder:
        causal = abs_k <= abs_q
        band = band & causal.unsqueeze(0).unsqueeze(0)

    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale
    mask_val = -1e4
    scores = scores.masked_fill(~band, mask_val)

    if token_valid is not None:
        kv_valid = token_valid[:, None, None, :].to(torch.bool)
        scores = scores.masked_fill(~kv_valid, mask_val)

    scores = torch.nan_to_num(scores, nan=mask_val, posinf=1e4, neginf=mask_val)

    combined = band if token_valid is None else (band & token_valid[:, None, None, :])
    any_valid = combined.any(dim=-1, keepdim=True).to(scores.dtype)

    attn = F.softmax(scores.float(), dim=-1).to(Q.dtype) * any_valid
    attn = F.dropout(attn, p=dropout_p, training=training)
    return torch.matmul(attn, V.to(attn.dtype)), attn


# ===========================================================================
# Global linear attention (SP-FIX-04, SP-FIX-17)
# ===========================================================================
# IMPORTANT: After low-rank projection, Q and K are (B, N, rank) — there is no
# "per-head" axis. V is (B, N, h*d). We compute the linear attention over the
# (B, N, rank) / (B, N, h*d) tensors and reshape at the very end.


def _causal_global_linear_attention(
    Q_feat: torch.Tensor,  # (B, N, rank)
    K_feat: torch.Tensor,  # (B, N, rank)
    V_flat: torch.Tensor,  # (B, N, h*d)
    chunk_size: int,
) -> torch.Tensor:
    """Chunked causal cumsum, memory O(B * chunk * rank * h*d)."""
    B, N, rank = Q_feat.shape
    D = V_flat.shape[-1]
    dtype = Q_feat.dtype
    device = Q_feat.device

    state_kv = torch.zeros(B, rank, D, device=device, dtype=dtype)
    state_k = torch.zeros(B, rank, device=device, dtype=dtype)
    outs = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        Qc = Q_feat[:, start:end]  # (B, C, rank)
        Kc = K_feat[:, start:end]  # (B, C, rank)
        Vc = V_flat[:, start:end]  # (B, C, D)

        # Local cumsum: (B, C, rank, D) via outer product then cumsum.
        KV_local = torch.einsum("bcr,bcd->bcrd", Kc, Vc).cumsum(dim=1)
        K_local = Kc.cumsum(dim=1)  # (B, C, rank)

        KV_total = KV_local + state_kv.unsqueeze(1)  # (B, C, rank, D)
        K_total = K_local + state_k.unsqueeze(1)  # (B, C, rank)

        num = torch.einsum("bcr,bcrd->bcd", Qc, KV_total)  # (B, C, D)
        den = torch.einsum("bcr,bcr->bc", Qc, K_total).unsqueeze(-1).clamp(min=1e-6)
        outs.append(num / den)

        state_kv = KV_total[:, -1]  # (B, rank, D)
        state_k = K_total[:, -1]  # (B, rank)

    return torch.cat(outs, dim=1)  # (B, N, D)


def _noncausal_global_linear_attention(
    Q_feat: torch.Tensor,  # (B, N, rank)
    K_feat: torch.Tensor,  # (B, N, rank)
    V_flat: torch.Tensor,  # (B, N, h*d)
) -> torch.Tensor:
    """Bidirectional linear attention via KV-first associativity."""
    # (B, rank, h*d)
    KV = torch.matmul(K_feat.transpose(-2, -1).contiguous(), V_flat)
    Z = K_feat.sum(dim=-2, keepdim=True)  # (B, 1, rank)
    num = torch.matmul(Q_feat, KV)  # (B, N, h*d)
    den = torch.matmul(Q_feat, Z.transpose(-2, -1).contiguous()).clamp(min=1e-6)
    return num / den


# ===========================================================================
# Linear state recurrence (SP-FIX-05)
# ===========================================================================


@torch.jit.script
def _state_recurrence_jit(
    u: torch.Tensor,
    gamma: torch.Tensor,
    reverse: bool = False,
) -> torch.Tensor:
    if reverse:
        u = torch.flip(u, dims=[1])
    gamma_b = gamma.view(1, 1, -1)
    B, N, D = u.shape
    state = torch.zeros(B, D, device=u.device, dtype=u.dtype)
    outs = []
    for t in range(N):
        state = gamma_b.squeeze(1) * state + u[:, t, :]
        outs.append(state)
    out = torch.stack(outs, dim=1)
    if reverse:
        out = torch.flip(out, dims=[1])
    return out


# ===========================================================================
# Main attention module
# ===========================================================================


@ATTENTION_REGISTRY.register("aegis_spectron_fp8", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("aegis_spectron_fp8_attn", supports_causal=True, supports_gqa=True)
class AegisSpectronFP8Attention(BaseAttention):
    is_custom_attention = True

    def __init__(
        self,
        config: Any | None = None,
        hidden_size: int | None = None,
        num_heads: int | None = None,
        pretrained_attn: nn.Module | None = None,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ):
        if not isinstance(config, AegisSpectronFP8Config):
            cfg_src = config if config is not None else kwargs
            if isinstance(cfg_src, dict):
                h_size = hidden_size or cfg_src.get("hidden_size", cfg_src.get("dim", 768))
                n_heads = num_heads or cfg_src.get(
                    "num_attention_heads", cfg_src.get("num_heads", 12)
                )
                n_kv = cfg_src.get("num_key_value_heads", n_heads)
                h_dim = cfg_src.get("head_dim", h_size // max(1, n_heads))
            else:
                h_size = hidden_size or kwargs.get(
                    "hidden_size",
                    getattr(cfg_src, "hidden_size", getattr(cfg_src, "d_model", 768)),
                )
                n_heads = num_heads or kwargs.get(
                    "num_heads",
                    getattr(
                        cfg_src,
                        "num_attention_heads",
                        getattr(cfg_src, "num_heads", 12),
                    ),
                )
                n_kv = getattr(
                    cfg_src,
                    "num_key_value_heads",
                    getattr(cfg_src, "num_kv_heads", n_heads),
                )
                h_dim = getattr(
                    cfg_src,
                    "head_dim",
                    getattr(cfg_src, "kv_channels", h_size // max(1, n_heads)),
                )
            config = AegisSpectronFP8Config(
                hidden_size=h_size,
                num_heads=n_heads,
                num_key_value_heads=n_kv,
                head_dim=h_dim,
                window_size=getattr(cfg_src, "window_size", getattr(cfg_src, "local_window", 64)),
                svd_rank=getattr(cfg_src, "svd_rank", getattr(cfg_src, "rank", min(128, h_size))),
                lora_rank=getattr(cfg_src, "lora_rank", getattr(cfg_src, "lora_r", 32)),
                dropout=getattr(cfg_src, "dropout", 0.1),
                fp8_format=getattr(cfg_src, "fp8_format", "e4m3"),
                is_decoder=_detect_decoder(cfg_src, is_decoder, pretrained_attn),
            )
        super().__init__(config)
        self.config: AegisSpectronFP8Config = config

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        if self.num_heads % max(1, self.num_kv_heads) != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})."
            )
        self.kv_repeat = self.num_heads // max(1, self.num_kv_heads)
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.supports_gqa = True
        self.is_nested_encoder = False
        self.is_decoder = _detect_decoder(config, is_decoder, pretrained_attn)
        self.fp8_format = config.fp8_format
        self.rank = config.svd_rank
        self.lora_rank = config.lora_rank

        self.router = SpectronSpectralEntropyRouter(config)

        self.q_proj = FactorizedLinear(
            self.hidden_size,
            self.all_head_size,
            rank=self.rank,
            lora_rank=self.lora_rank,
            bias=True,
        )
        self.k_proj = FactorizedLinear(
            self.hidden_size,
            self.all_kv_head_size,
            rank=self.rank,
            lora_rank=self.lora_rank,
            bias=True,
        )
        self.v_proj = FactorizedLinear(
            self.hidden_size,
            self.all_kv_head_size,
            rank=self.rank,
            lora_rank=self.lora_rank,
            bias=True,
        )
        self.out_proj = FactorizedLinear(
            self.all_head_size,
            self.hidden_size,
            rank=self.rank,
            lora_rank=self.lora_rank,
            bias=True,
        )

        # SP-FIX-15: separate Q/K low-rank projections.
        self.svd_U_q = nn.Parameter(torch.randn(self.all_head_size, self.rank) * 0.02)
        self.svd_U_k = nn.Parameter(torch.randn(self.all_head_size, self.rank) * 0.02)

        # Linear recurrence parameters (SP-FIX-06: no dead linear_A).
        self.linear_B = nn.Parameter(torch.ones(self.all_head_size))
        self.linear_C = nn.Parameter(torch.ones(self.all_head_size))
        self.decay_param = nn.Parameter(torch.full((self.all_head_size,), 4.5951))
        # SP-FIX-13: per-head gate.
        self.bidir_gate = nn.Parameter(torch.zeros(self.num_heads))
        self.linear_in = (
            nn.Linear(self.hidden_size, self.all_head_size, bias=False)
            if self.hidden_size != self.all_head_size
            else None
        )

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------
    @property
    def query(self):
        return self.q_proj

    @property
    def key(self):
        return self.k_proj

    @property
    def value(self):
        return self.v_proj

    @property
    def q_lin(self):
        return self.q_proj

    @property
    def k_lin(self):
        return self.k_proj

    @property
    def v_lin(self):
        return self.v_proj

    @property
    def out_lin(self):
        return self.out_proj

    @property
    def dense(self):
        return self.out_proj

    @property
    def o_proj(self):
        return self.out_proj

    @property
    def c_proj(self):
        return self.out_proj

    @property
    def q(self):
        return self.q_proj

    @property
    def k(self):
        return self.k_proj

    @property
    def v(self):
        return self.v_proj

    @property
    def o(self):
        return self.out_proj

    # ------------------------------------------------------------------
    # SVD init of FactorizedLinear
    # ------------------------------------------------------------------
    @staticmethod
    def _svd_init(w: torch.Tensor, b: torch.Tensor | None, proj: FactorizedLinear) -> None:
        with torch.no_grad():
            W = w.detach().float()
            if W.ndim != 2:
                raise ValueError(f"Expected 2D weight, got shape {tuple(W.shape)}")
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            r = min(proj.rank, U.size(1), Vh.size(0))
            proj.lin1.weight.data.zero_()
            proj.lin2.weight.data.zero_()
            in_copy = min(W.shape[1], proj.lin1.weight.size(1))
            out_copy = min(W.shape[0], proj.lin2.weight.size(0))
            proj.lin1.weight.data[:r, :in_copy].copy_(Vh[:r, :in_copy].to(proj.lin1.weight.dtype))
            US = U[:out_copy, :r] @ torch.diag(S[:r])
            proj.lin2.weight.data[:out_copy, :r].copy_(US.to(proj.lin2.weight.dtype))
            if b is not None and proj.lin2.bias is not None:
                n = min(b.numel(), proj.lin2.bias.numel())
                proj.lin2.bias.data[:n].copy_(b.detach()[:n].to(proj.lin2.bias.dtype))
            elif proj.lin2.bias is not None:
                proj.lin2.bias.data.zero_()
            proj.lin1.weight.requires_grad = False
            proj.lin2.weight.requires_grad = False
            if proj.lin2.bias is not None:
                proj.lin2.bias.requires_grad = False
            if proj.lora_A is not None:
                nn.init.kaiming_uniform_(proj.lora_A, a=math.sqrt(5))
                nn.init.zeros_(proj.lora_B)

    # ------------------------------------------------------------------
    # Weight extraction (universal across architectures)
    # ------------------------------------------------------------------
    def _extract_proj_weights(
        self, old_attn: nn.Module
    ) -> dict[str, tuple[torch.Tensor | None, torch.Tensor | None]]:
        from neuralops.core.adapters import UniversalAdapter

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2 fused c_attn
        if hasattr(old_attn, "c_attn"):
            mod = old_attn.c_attn
            if hasattr(mod, "merge") and not getattr(mod, "merged", False):
                try:
                    mod.merge()
                except Exception:
                    pass
            base = getattr(mod, "base_layer", mod)
            c_w = getattr(base, "weight", getattr(mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data
                is_conv1d = (
                    mod.__class__.__name__ == "Conv1D"
                    or base.__class__.__name__ == "Conv1D"
                    or hasattr(mod, "nf")
                    or hasattr(base, "nf")
                )
                if is_conv1d or (c_w.dim() == 2 and c_w.shape[1] == 3 * self.hidden_size):
                    c_w = c_w.t().contiguous()
                split = c_w.size(0) // 3
                bias_holder = getattr(base, "bias", getattr(mod, "bias", None))
                c_b = bias_holder.data if bias_holder is not None else None
                for i, p in enumerate(("q", "k", "v")):
                    extracted[p] = (
                        c_w[i * split : (i + 1) * split, :],
                        c_b[i * split : (i + 1) * split] if c_b is not None else None,
                    )

        # 2. GPT-2 c_proj
        if hasattr(old_attn, "c_proj"):
            mod = old_attn.c_proj
            if hasattr(mod, "merge") and not getattr(mod, "merged", False):
                try:
                    mod.merge()
                except Exception:
                    pass
            base = getattr(mod, "base_layer", mod)
            w = getattr(base, "weight", getattr(mod, "weight", None))
            if w is not None:
                w = w.data
                is_conv1d = (
                    mod.__class__.__name__ == "Conv1D"
                    or base.__class__.__name__ == "Conv1D"
                    or hasattr(mod, "nf")
                    or hasattr(base, "nf")
                )
                if is_conv1d or (
                    w.dim() == 2
                    and w.shape[0] != self.hidden_size
                    and w.shape[1] == self.hidden_size
                ):
                    w = w.t().contiguous()
                b = getattr(base, "bias", getattr(mod, "bias", None))
                extracted["o"] = (w, b.data if b is not None else None)

        # 3. BERT / RoBERTa nested encoder
        if hasattr(old_attn, "self"):
            s = old_attn.self
            if hasattr(s, "query") and extracted["q"][0] is None:
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key") and extracted["k"][0] is None:
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value") and extracted["v"][0] is None:
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        # 4. BERT output.dense
        if (
            hasattr(old_attn, "output")
            and hasattr(old_attn.output, "dense")
            and extracted["o"][0] is None
        ):
            d = old_attn.output.dense
            extracted["o"] = (d.weight.data, getattr(d, "bias", None))

        # 5. Named projections
        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "v": ["v_proj", "v_lin", "value"],
            "o": ["o_proj", "out_proj", "out_lin", "dense"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for n in names:
                    if hasattr(old_attn, n):
                        m = getattr(old_attn, n)
                        if hasattr(m, "weight"):
                            extracted[proj] = (m.weight.data, getattr(m, "bias", None))
                            break

        # 6. UniversalAdapter fallback
        for p in ("q", "k", "v", "o"):
            if extracted[p][0] is None:
                try:
                    w, b = UniversalAdapter.get_weight_bias(old_attn, p)
                    if w is not None:
                        extracted[p] = (w, b)
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed for %s: %s", p, e)

        return extracted

    def copy_weights(self, old_attn: nn.Module) -> None:
        extracted = self._extract_proj_weights(old_attn)

        cls_name = old_attn.__class__.__name__
        has_out = any(
            hasattr(old_attn, a)
            for a in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o")
        )
        if (
            hasattr(old_attn, "out_lin")
            or hasattr(old_attn, "c_proj")
            or hasattr(old_attn, "o_proj")
        ):
            self.is_nested_encoder = False
        elif (
            (
                hasattr(old_attn, "self")
                and hasattr(old_attn.self, "query")
                and hasattr(old_attn, "output")
                and hasattr(old_attn.output, "dense")
            )
            or cls_name
            in (
                "BertSelfAttention",
                "RobertaSelfAttention",
                "XLMRobertaSelfAttention",
                "ElectraSelfAttention",
            )
            or (cls_name.endswith("SelfAttention") and not has_out)
        ):
            self.is_nested_encoder = True
        else:
            self.is_nested_encoder = False

        missing = [p for p in ("q", "k", "v") if extracted[p][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(
                f"AegisSpectronFP8Attention.copy_weights: missing {missing} from "
                f"{cls_name}. Refusing random init."
            )

        for proj, target in (
            ("q", self.q_proj),
            ("k", self.k_proj),
            ("v", self.v_proj),
        ):
            w, b = extracted[proj]
            if w is not None:
                self._svd_init(w, b, target)

        if self.is_nested_encoder:
            with torch.no_grad():
                self.out_proj.lin1.weight.data.zero_()
                self.out_proj.lin2.weight.data.zero_()
                r = min(self.out_proj.rank, self.hidden_size, self.all_head_size)
                self.out_proj.lin1.weight.data[:r, :r].fill_diagonal_(1.0)
                self.out_proj.lin2.weight.data[:r, :r].fill_diagonal_(1.0)
                if self.out_proj.lin2.bias is not None:
                    self.out_proj.lin2.bias.data.zero_()
        elif extracted["o"][0] is not None:
            self._svd_init(extracted["o"][0], extracted["o"][1], self.out_proj)

        # Init svd_U_q / svd_U_k from Q / K weights.
        with torch.no_grad():
            if extracted["q"][0] is not None:
                U, _, _ = torch.linalg.svd(extracted["q"][0].float(), full_matrices=False)
                r = min(self.rank, U.size(1), self.svd_U_q.size(1))
                self.svd_U_q.data[:, :r].copy_(U[: self.svd_U_q.size(0), :r].to(self.svd_U_q.dtype))
            if extracted["k"][0] is not None:
                U, _, _ = torch.linalg.svd(extracted["k"][0].float(), full_matrices=False)
                r = min(self.rank, U.size(1), self.svd_U_k.size(1))
                self.svd_U_k.data[:, :r].copy_(U[: self.svd_U_k.size(0), :r].to(self.svd_U_k.dtype))

        if hasattr(old_attn, "is_decoder"):
            self.is_decoder = bool(old_attn.is_decoder)
        elif hasattr(old_attn, "c_attn") or "gpt" in old_attn.__class__.__name__.lower():
            self.is_decoder = True

    # ------------------------------------------------------------------
    # Mask helper
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_valid_mask(
        attention_mask: torch.Tensor | None,
        B: int,
        N_kv: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            if attention_mask.dtype == torch.bool:
                return attention_mask
            if attention_mask.dtype.is_floating_point:
                return attention_mask > -1e4
            return attention_mask > 0
        if attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                v = attention_mask
            elif attention_mask.dtype.is_floating_point:
                v = attention_mask > -1e4
            else:
                v = attention_mask > 0
            return v.any(dim=-2).squeeze(1)
        raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")

    # ------------------------------------------------------------------
    # Branch: local
    # ------------------------------------------------------------------
    def _local_branch(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        token_valid: torch.Tensor | None,
        past_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _local_window_attention(
            Q,
            K,
            V,
            window=self.config.window_size,
            token_valid=token_valid,
            is_decoder=self.is_decoder,
            past_len=past_len,
            dropout_p=self.config.dropout,
            training=self.training,
        )

    # ------------------------------------------------------------------
    # Branch: global (SP-FIX-17 — correct shapes)
    # ------------------------------------------------------------------
    def _global_branch(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        token_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Returns context with the same shape as Q: (B, h, N, d).

        The low-rank projection collapses the head axis, so all linear
        attention math happens on (B, N, rank) / (B, N, h*d) tensors.
        Head structure is restored at the very end via a final view.
        """
        B, h, N, d = Q.shape
        D = h * d

        # Flatten heads → (B, N, h*d)
        Q_flat = Q.transpose(1, 2).contiguous().view(B, N, D)
        K_flat = K.transpose(1, 2).contiguous().view(B, N, D)
        V_flat = V.transpose(1, 2).contiguous().view(B, N, D)

        # Low-rank projection → (B, N, rank)
        Q_lr = torch.matmul(Q_flat, self.svd_U_q.to(Q_flat.dtype))
        K_lr = torch.matmul(K_flat, self.svd_U_k.to(K_flat.dtype))

        # Non-negative kernel feature map φ(x) = ELU(x) + 1
        phi_Q = F.elu(Q_lr) + 1.0  # (B, N, rank)
        phi_K = F.elu(K_lr) + 1.0

        # Run in fp32 for numerical stability (normalization can overflow in fp16).
        with torch.autocast(device_type=Q.device.type, enabled=False):
            phi_Q_f = phi_Q.float()
            phi_K_f = phi_K.float()
            V_f = V_flat.float()

            # SP-FIX-02: mask padding keys.
            if token_valid is not None:
                m = token_valid[:, :, None].to(phi_K_f.dtype)  # (B, N, 1)
                phi_K_f = phi_K_f * m
                V_f = V_f * m

            if self.is_decoder:
                out = _causal_global_linear_attention(
                    phi_Q_f,
                    phi_K_f,
                    V_f,
                    chunk_size=self.config.chunk_size,
                )
            else:
                out = _noncausal_global_linear_attention(phi_Q_f, phi_K_f, V_f)

            # (B, N, h*d) → (B, h, N, d)
            return (
                torch.nan_to_num(out, nan=0.0)
                .to(Q.dtype)
                .view(B, N, h, d)
                .transpose(1, 2)
                .contiguous()
            )

    # ------------------------------------------------------------------
    # Branch: linear recurrence (SP-FIX-05)
    # ------------------------------------------------------------------
    def _linear_branch(
        self,
        X: torch.Tensor,
        token_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.linear_in is not None:
            X = self.linear_in(X)
        B, N, D = X.shape

        if token_valid is not None:
            X = X * token_valid.unsqueeze(-1).to(X.dtype)

        u = X * self.linear_B.view(1, 1, D).to(X.dtype)
        gamma = torch.sigmoid(self.decay_param.to(X.dtype)).clamp(min=1e-4, max=1.0 - 1e-4)

        out_fwd = _state_recurrence_jit(u, gamma, reverse=False)

        if self.is_decoder:
            out_h = out_fwd.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            out_bwd = _state_recurrence_jit(u, gamma, reverse=True)
            gate = torch.sigmoid(self.bidir_gate.to(X.dtype)).view(1, self.num_heads, 1, 1)
            out_fwd_h = out_fwd.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            out_bwd_h = out_bwd.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            out_h = gate * out_fwd_h + (1.0 - gate) * out_bwd_h

        out_h = out_h * self.linear_C.view(1, self.num_heads, 1, self.head_dim).to(X.dtype)
        return out_h.contiguous()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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
    ):
        B, N, _ = hidden_states.shape
        device = hidden_states.device

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv_new = kv_input.shape[1]

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        token_valid_new = self._prepare_valid_mask(eff_mask, B, N_kv_new, device)

        # Routing on hidden states (self-attention only).
        alpha = self.router(hidden_states, token_valid=token_valid_new)
        alpha_loc = alpha[:, :, 0].view(B, 1, N, 1)
        alpha_glob = alpha[:, :, 1].view(B, 1, N, 1)
        alpha_lin = alpha[:, :, 2].view(B, 1, N, 1)

        # FP8 on projected activations (SP-FIX-01).
        fp8_fmt = self.fp8_format
        use_ste = self.training

        Q_new = fp8_cast_simulate(self.q_proj(hidden_states), format=fp8_fmt, use_ste=use_ste)
        K_new = fp8_cast_simulate(self.k_proj(kv_input), format=fp8_fmt, use_ste=use_ste)
        V_new = fp8_cast_simulate(self.v_proj(kv_input), format=fp8_fmt, use_ste=use_ste)

        Q = Q_new.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        K_new_h = (
            K_new.view(B, N_kv_new, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        )
        V_new_h = (
            V_new.view(B, N_kv_new, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        )

        # RoPE with past_len offset (SP-FIX-09).
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            past_len_rope = 0
            if past_key_value is not None:
                if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 1:
                    past_len_rope = past_key_value[0].shape[-2]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    past_len_rope = past_key_value.key_cache[0].shape[-2]
            cos = cos[..., past_len_rope : past_len_rope + N, :]
            sin = sin[..., past_len_rope : past_len_rope + N, :]
            Q, K_new_h = self._apply_rope(Q, K_new_h, cos, sin)

        # KV cache (SP-FIX-03).
        past_k = past_v = None
        if past_key_value is not None and not is_cross:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
                past_k, past_v = past_key_value[0], past_key_value[1]
            elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                idx = min(getattr(self, "layer_idx", 0), len(past_key_value.key_cache) - 1)
                past_k = past_key_value.key_cache[idx]
                past_v = past_key_value.value_cache[idx]

        if past_k is not None and not is_cross:
            K_cat = torch.cat([past_k, K_new_h], dim=-2)
            V_cat = torch.cat([past_v, V_new_h], dim=-2)
        else:
            K_cat = K_new_h
            V_cat = V_new_h

        if past_key_value is not None and hasattr(past_key_value, "update") and not is_cross:
            idx = getattr(self, "layer_idx", 0)
            past_key_value.update(K_new_h, V_new_h, idx)

        N_kv = K_cat.shape[2]
        past_len = N_kv - N if (self.is_decoder and not is_cross) else 0

        if past_len > 0 and token_valid_new is not None:
            past_valid = torch.ones(B, past_len, device=device, dtype=torch.bool)
            token_valid = torch.cat([past_valid, token_valid_new], dim=1)
        else:
            token_valid = token_valid_new

        K_full = repeat_kv(K_cat, self.kv_repeat) if self.kv_repeat > 1 else K_cat
        V_full = repeat_kv(V_cat, self.kv_repeat) if self.kv_repeat > 1 else V_cat

        out_loc, attn_loc = self._local_branch(Q, K_full, V_full, token_valid, past_len)
        out_glob = self._global_branch(Q, K_full, V_full, token_valid)
        out_lin = self._linear_branch(hidden_states, token_valid_new)
        target_dt = hidden_states.dtype
        fused = (
            alpha_loc.to(target_dt) * out_loc.to(target_dt)
            + alpha_glob.to(target_dt) * out_glob.to(target_dt)
            + alpha_lin.to(target_dt) * out_lin.to(target_dt)
        )

        # head_mask (SP-FIX-03).
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask[:, :, :1].unsqueeze(-1)
            fused = fused * head_mask.to(fused.dtype)

        out = fused.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            final = out
        else:
            final = self.out_proj(out)
        final = torch.nan_to_num(final, nan=0.0, posinf=1e4, neginf=-1e4)

        attn_ret = attn_loc if output_attentions else None
        return final.to(hidden_states.dtype), attn_ret

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_rope(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def _rotate_half(x: torch.Tensor) -> torch.Tensor:
            d = x.shape[-1]
            h = d // 2
            rot = torch.cat((-x[..., h : 2 * h], x[..., :h]), dim=-1)
            if d % 2 == 1:
                rot = torch.cat([rot, x[..., -1:]], dim=-1)
            return rot

        if cos.dim() == 3:
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        while cos.dim() < q.dim():
            cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)

        if cos.shape[-2] != q.shape[-2]:
            raise ValueError(f"RoPE seq len mismatch: cos {cos.shape[-2]} vs q {q.shape[-2]}")

        rotary_dim = cos.shape[-1]
        head_dim = q.shape[-1]
        if rotary_dim <= head_dim:
            q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
            k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
            q_rot = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
            k_rot = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
            q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot
            k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot
        else:
            cos_cut, sin_cut = cos[..., :head_dim], sin[..., :head_dim]
            q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
            k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))
        return q, k

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.zeros((), device=self.out_proj.lin1.weight.device)


# ===========================================================================
# Injection
# ===========================================================================


@ATTENTION_REGISTRY.register("aegis_spectron_fp8_inject")
@ATTENTION_REGISTRY.register("inject_aegis_spectron_fp8")
def inject_aegis_spectron_fp8(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_aegis_spectron_fp8: max recursion depth.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg_root = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal.
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif (
        hasattr(target, "bert")
        and hasattr(target.bert, "encoder")
        and hasattr(target.bert.encoder, "layer")
    ):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        n_inj = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_aegis_spectron_fp8(
                layer=lyr,
                model_config=cfg_root,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                n_inj += 1
        return target if n_inj > 0 else None

    layer = target
    try:
        if old_attn is None:
            for attr in ("attention", "attn", "self_attn", "self"):
                if hasattr(layer, attr):
                    cand = getattr(layer, attr)
                    if cand is not None:
                        old_attn = getattr(cand, "self", cand)
                        break

        if model_config is None:
            if any(hasattr(layer, a) for a in ("hidden_size", "d_model", "n_embd", "dim")):
                model_config = layer
            elif old_attn is not None:
                model_config = getattr(old_attn, "config", None)

        h_size = (
            getattr(
                model_config,
                "hidden_size",
                getattr(
                    model_config,
                    "d_model",
                    getattr(model_config, "n_embd", getattr(model_config, "dim", 768)),
                ),
            )
            if model_config is not None
            else 768
        )
        n_heads = (
            getattr(
                model_config,
                "num_attention_heads",
                getattr(
                    model_config,
                    "num_heads",
                    getattr(model_config, "n_heads", getattr(model_config, "n_head", 12)),
                ),
            )
            if model_config is not None
            else 12
        )
        h_dim = (
            getattr(
                model_config,
                "head_dim",
                getattr(model_config, "kv_channels", h_size // max(1, n_heads)),
            )
            if model_config is not None
            else h_size // max(1, n_heads)
        )
        n_kv_heads = (
            getattr(
                model_config,
                "num_key_value_heads",
                getattr(model_config, "num_kv_heads", n_heads),
            )
            if model_config is not None
            else n_heads
        )

        is_dec = _detect_decoder(model_config, kwargs.get("is_decoder"), old_attn)

        cfg = AegisSpectronFP8Config(
            hidden_size=h_size,
            num_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            head_dim=h_dim,
            window_size=(
                getattr(comp_config, "window_size", getattr(comp_config, "local_window", 64))
                if comp_config
                else 64
            ),
            svd_rank=(
                getattr(comp_config, "svd_rank", getattr(comp_config, "rank", 128))
                if comp_config
                else 128
            ),
            lora_rank=(
                getattr(comp_config, "lora_rank", getattr(comp_config, "lora_r", 32))
                if comp_config
                else 32
            ),
            dropout=getattr(comp_config, "dropout", 0.1) if comp_config else 0.1,
            is_decoder=is_dec,
            fp8_format=(getattr(comp_config, "fp8_format", "e4m3") if comp_config else "e4m3"),
            chunk_size=getattr(comp_config, "chunk_size", 64) if comp_config else 64,
        )

        try:
            new_attn = AegisSpectronFP8Attention(config=cfg, pretrained_attn=old_attn)
        except Exception as e:
            logger.error("inject_aegis_spectron_fp8: construction failed: %s", e)
            return None

        if old_attn is not None and hasattr(old_attn, "rotary_emb"):
            new_attn.rotary_emb = old_attn.rotary_emb

        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                new_attn.is_nested_encoder = True
                layer.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                layer.attention = new_attn
        elif hasattr(layer, "self_attn"):
            new_attn.is_nested_encoder = False
            layer.self_attn = new_attn
        elif hasattr(layer, "self"):
            new_attn.is_nested_encoder = False
            layer.self = new_attn
        elif hasattr(layer, "attn"):
            new_attn.is_nested_encoder = False
            layer.attn = new_attn
        return new_attn
    except Exception as e:
        logger.exception("Failed to inject AEGIS-Spectron-FP8: %s", e)
        return None
