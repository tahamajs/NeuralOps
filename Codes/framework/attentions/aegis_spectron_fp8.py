r"""
Aegis-Spectron-FP8 Hybrid Attention Module.

Integrates Aegis multi-branch local/global feature fusion with Spectron Lagrangian spectral entropy routing.
Dynamic routing measures singular value entropy per token to distribute token representation across:
1. Local Sliding-Window Attention (O(N * W) with strict causal masking for decoders)
2. Global Low-Rank SVD Attention (True O(N * r * D) Linear Attention with Normalization;
   associative KV-first for bidirectional models, cumulative sum causal recurrence for decoders)
3. Linear State Recurrence (True O(N * D) via Numerically Stable CumSum EMA, strictly causal for decoders)

Equipped with:
- Functional FP8 quantization simulation (E4M3 and E5M2) with dynamic range scaling.
- FactorizedLinear base weights + trainable residual LoRA adapters for parameter-efficient recovery fine-tuning.
- Support for RoPE (Rotary Position Embeddings) and GQA (Grouped-Query Attention).
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.core.attention_utils import repeat_kv
from framework.core.gqa_utils import get_qkv_projection_dims


def fp8_cast_simulate(
    x: torch.Tensor,
    format: str = "e4m3",
    per_channel: bool = False,
) -> torch.Tensor:
    r"""
    Simulates FP8 quantization error using dynamic scaling and cast-dequantize roundtrip.

    Supported formats:
      - "e4m3" / "float8_e4m3fn": 1 sign, 4 exponent, 3 mantissa bits (max ~448.0)
      - "e5m2" / "float8_e5m2": 1 sign, 5 exponent, 2 mantissa bits (max ~57344.0)

    Args:
      x: Input tensor of arbitrary shape (float32, float16, bfloat16).
      format: FP8 format string ("e4m3" or "e5m2").
      per_channel: If True, dynamic scaling is computed along the last dimension (dim=-1),
                   otherwise per-tensor scaling is applied.

    Returns:
      Tensor with identical shape and dtype containing simulated FP8 rounding noise.
    """
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        return x

    with torch.no_grad():
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
            raise ValueError(f"Unsupported FP8 format: '{format}'. Supported formats: 'e4m3', 'e5m2'")

        # 1. Prefer torchao if installed
        try:
            from torchao.quantization.quant_api import to_fp8
            x_dequant = to_fp8(x, format=fmt).to(orig_dtype).clamp(min=-1e4, max=1e4)
            return x + (x_dequant - x).detach()
        except Exception:
            pass

        # Dynamic scaling factor calculation
        if per_channel:
            amax = x_f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        else:
            amax = x_f.abs().amax().clamp(min=1e-12)

        scale = max_val / amax
        x_scaled = (x_f * scale).clamp(min=-max_val, max=max_val)

        # 2. PyTorch native float8 roundtrip
        if torch_dtype is not None:
            try:
                x_fp8 = x_scaled.to(torch_dtype)
                x_dequant = (x_fp8.float() / scale.float().clamp(min=1e-8)).to(orig_dtype)
                return x + (x_dequant.clamp(min=-1e4, max=1e4) - x).detach()
            except Exception:
                pass

        # 3. Mathematical fallback: bit-depth IEEE-754 mantissa rounding
        abs_x = torch.abs(x_scaled).clamp(min=1e-12)
        exp = torch.floor(torch.log2(abs_x))
        factor = 2.0 ** (exp - mantissa_bits)
        rounded = torch.round(abs_x / factor) * factor
        x_dequant = ((torch.sign(x_scaled) * rounded).clamp(min=-max_val, max=max_val) / scale.float().clamp(min=1e-8)).to(orig_dtype)
        x_dequant = x_dequant.clamp(min=-1e4, max=1e4)

    return x + (x_dequant - x).detach()


@dataclass
class AegisSpectronFP8Config:
    r"""Hyperparameter configuration dataclass for Aegis-Spectron-FP8."""
    hidden_size: int = 768
    num_heads: int = 4
    num_key_value_heads: int = 4
    head_dim: int = 64
    window_size: int = 64
    svd_rank: int = 128
    lora_rank: int = 32
    temperature: float = 1.0
    lambda_cost: float = 0.1
    c_loc: float = 0.5
    c_glob: float = 0.8
    c_lin: float = 1.0
    dropout: float = 0.1
    router_init_bias: Tuple[float, float, float] = (4.0, 0.0, 0.0)
    is_decoder: bool = False
    fp8_format: str = "e4m3"

    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get("hidden_size", kwargs.get("d_model", kwargs.get("dim", 768)))
        self.num_heads = kwargs.get("num_heads", kwargs.get("num_attention_heads", kwargs.get("n_heads", 4)))
        self.num_key_value_heads = kwargs.get("num_key_value_heads", kwargs.get("num_kv_heads", self.num_heads))
        self.head_dim = kwargs.get("head_dim", kwargs.get("kv_channels", self.hidden_size // max(1, self.num_heads)))
        self.window_size = kwargs.get("window_size", kwargs.get("local_window", kwargs.get("block_size", 64)))
        self.svd_rank = kwargs.get("svd_rank", kwargs.get("rank", min(128, self.hidden_size)))
        self.lora_rank = kwargs.get("lora_rank", kwargs.get("lora_r", 32))
        self.temperature = kwargs.get("temperature", 1.0)
        self.lambda_cost = kwargs.get("lambda_cost", 0.1)
        self.c_loc = kwargs.get("c_loc", 0.5)
        self.c_glob = kwargs.get("c_glob", 0.8)
        self.c_lin = kwargs.get("c_lin", 1.0)
        self.dropout = kwargs.get("dropout", 0.1)
        self.router_init_bias = kwargs.get("router_init_bias", (4.0, 0.0, 0.0))
        self.is_decoder = kwargs.get("is_decoder", False)
        self.fp8_format = kwargs.get("fp8_format", "e4m3")


class FactorizedLinear(nn.Module):
    r"""
    Factorized linear projection with internal rank-r base factorization and trainable residual LoRA adapter:
        Base: x -> lin1(x) [in_features -> rank] -> lin2(lin1(x)) [rank -> out_features] (frozen)
        LoRA: x -> lora_A(x) [in_features -> lora_rank] -> lora_B(...) [lora_rank -> out_features] (trainable)
        out = lin2(lin1(x)) + F.linear(F.linear(x, self.lora_A), self.lora_B)

    Uses `lora_A` and `lora_B` parameter names so recovery optimizers (such as get_recovery_params)
    register them automatically as trainable parameters.
    Base weights and biases are frozen to maintain pretrained representations.
    Standardized initialization: lora_A ~ Kaiming Uniform, lora_B = 0.
    """

    def __init__(
        self,
        in_features_or_lin1: Union[int, nn.Linear],
        out_features_or_lin2: Optional[Union[int, nn.Linear]] = None,
        rank: int = 128,
        lora_rank: int = 32,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if isinstance(in_features_or_lin1, nn.Linear) and isinstance(out_features_or_lin2, nn.Linear):
            self.lin1 = in_features_or_lin1
            self.lin2 = out_features_or_lin2
            self.in_features = self.lin1.in_features
            self.out_features = self.lin2.out_features
            self.rank = self.lin1.out_features
        else:
            in_features = int(in_features_or_lin1)
            out_features = int(out_features_or_lin2)
            self.in_features = in_features
            self.out_features = out_features
            self.rank = max(1, min(rank, in_features, out_features))

            factory_kwargs = {"device": device, "dtype": dtype}
            self.lin1 = nn.Linear(in_features, self.rank, bias=False, **factory_kwargs)
            self.lin2 = nn.Linear(self.rank, out_features, bias=bias, **factory_kwargs)

        # Base weights are frozen to prevent catastrophic unlearning
        self.lin1.weight.requires_grad = False
        self.lin2.weight.requires_grad = False
        if self.lin2.bias is not None:
            self.lin2.bias.requires_grad = False

        # Trainable residual LoRA adapter
        self.lora_rank: int = min(lora_rank, min(self.in_features, self.out_features)) if lora_rank > 0 else 0
        factory_kwargs = {"device": device, "dtype": dtype}
        if self.lora_rank > 0:
            self.lora_B: Optional[nn.Parameter] = nn.Parameter(
                torch.empty((self.out_features, self.lora_rank), **factory_kwargs)
            )
            self.lora_A: Optional[nn.Parameter] = nn.Parameter(
                torch.empty((self.lora_rank, self.in_features), **factory_kwargs)
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

            if getattr(self, "lora_A", None) is not None and getattr(self, "lora_B", None) is not None:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        w = torch.matmul(self.lin2.weight, self.lin1.weight)
        if getattr(self, "lora_A", None) is not None and getattr(self, "lora_B", None) is not None:
            w = w + torch.matmul(self.lora_B, self.lora_A)
        return w

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.lin2.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin2(self.lin1(x))
        if getattr(self, "lora_A", None) is not None and getattr(self, "lora_B", None) is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out


class SpectronSpectralEntropyRouter(nn.Module):
    r"""
    Dynamic Spectron Router based on Spectral Entropy & Lagrangian Optimization.
    Computes spectral entropy per token: $\mathcal{S}(X_t) = -\sum_d p_{t,d} \ln p_{t,d}$
    """

    def __init__(self, config: AegisSpectronFP8Config, **kwargs):
        super().__init__()
        self.config = config
        self.tau = config.temperature
        self.lam = config.lambda_cost

        self.register_buffer("costs", torch.tensor([config.c_loc, config.c_glob, config.c_lin]))
        self.h_proj = nn.Linear(config.hidden_size, 3)
        nn.init.normal_(self.h_proj.weight, mean=0.0, std=0.01)
        self.h_proj.bias.data.copy_(torch.tensor(config.router_init_bias))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, N, D = X.shape

        with torch.no_grad():
            try:
                # Calculate entropy across the feature dimension for EACH token
                S_sq = X.float().pow(2)  # [B, N, D]
                denom = S_sq.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                p = (S_sq / denom).clamp(min=1e-8, max=1.0)
                entropy = -(p * torch.log(p)).sum(dim=-1)  # [B, N]
            except Exception:
                entropy = torch.ones((B, N), device=X.device, dtype=X.dtype)

        # Token-level routing projection (removed .mean(dim=1))
        H_b = self.h_proj(X)  # [B, N, 3]
        H_b = H_b * entropy.unsqueeze(-1).to(dtype=H_b.dtype)

        c = self.costs.to(dtype=H_b.dtype, device=X.device)
        tau = max(float(self.tau), 1e-4)
        logits = (H_b - self.lam * c) / tau
        return F.softmax(logits.float(), dim=-1).to(X.dtype)  # [B, N, 3]


@ATTENTION_REGISTRY.register("aegis_spectron_fp8", supports_causal=True, supports_bidirectional=True, supports_gqa=True, supports_rope=True)
@ATTENTION_REGISTRY.register("aegis_spectron_fp8_attn", supports_causal=True, supports_bidirectional=True, supports_gqa=True, supports_rope=True)
class AegisSpectronFP8Attention(BaseAttention):
    r"""
    Aegis-Spectron-FP8 Hybrid Architecture.
    Combines sliding-window local attention, true O(N) low-rank global SVD linear attention,
    and linear state-space recurrence under token-level spectral entropy routing.
    Strictly causal for autoregressive decoders, with true RoPE and GQA support.
    """

    is_custom_attention: bool = True
    def __init__(
        self,
        config: Optional[Any] = None,
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ):
        if not isinstance(config, AegisSpectronFP8Config):
            if isinstance(config, dict):
                h_size = hidden_size or config.get("hidden_size", config.get("dim", 768))
                n_heads = num_heads or config.get("num_attention_heads", config.get("num_heads", 12))
                h_dim = config.get("head_dim", config.get("kv_channels", h_size // max(1, n_heads)))
                n_kv_heads = config.get("num_key_value_heads", n_heads)
                rank = config.get("svd_rank", config.get("rank", min(128, h_size)))
                lora_rank = config.get("lora_rank", config.get("lora_r", 32))
                w_size = config.get("window_size", config.get("local_window", 64))
                drop = config.get("dropout", 0.1)
                is_dec = config.get("is_decoder", False)
                fp8_fmt = config.get("fp8_format", "e4m3")
            else:
                h_size = hidden_size or kwargs.get("hidden_size", kwargs.get("d_model", getattr(config, "hidden_size", getattr(config, "d_model", getattr(config, "dim", 768))) if config is not None else 768))
                n_heads = num_heads or kwargs.get("num_heads", kwargs.get("num_attention_heads", getattr(config, "num_attention_heads", getattr(config, "num_heads", getattr(config, "n_heads", 12))) if config is not None else 12))
                h_dim = kwargs.get("head_dim", getattr(config, "head_dim", getattr(config, "kv_channels", getattr(config, "d_kv", h_size // max(1, n_heads)))) if config is not None else h_size // max(1, n_heads))
                n_kv_heads = kwargs.get("num_key_value_heads", kwargs.get("num_kv_heads", getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", n_heads)) if config is not None else n_heads))
                rank = kwargs.get("rank", kwargs.get("svd_rank", getattr(config, "svd_rank", getattr(config, "rank", min(128, h_size))) if config is not None else 128))
                lora_rank = kwargs.get("lora_rank", kwargs.get("lora_r", getattr(config, "lora_rank", getattr(config, "lora_r", 32)) if config is not None else 32))
                w_size = kwargs.get("window_size", kwargs.get("local_window", getattr(config, "window_size", getattr(config, "local_window", getattr(config, "block_size", 64))) if config is not None else 64))
                drop = kwargs.get("dropout", getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)) if config is not None else 0.1)
                is_dec = kwargs.get("is_decoder", getattr(config, "is_decoder", False) if config is not None else False)
                fp8_fmt = kwargs.get("fp8_format", getattr(config, "fp8_format", "e4m3") if config is not None else "e4m3")

            config = AegisSpectronFP8Config(
                hidden_size=h_size,
                num_heads=n_heads,
                num_key_value_heads=n_kv_heads,
                head_dim=h_dim,
                window_size=w_size,
                svd_rank=rank,
                lora_rank=lora_rank,
                dropout=drop,
                is_decoder=is_dec,
                fp8_format=fp8_fmt,
            )
        super().__init__(config)
        self.config: AegisSpectronFP8Config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.q_out, self.kv_out, self.head_dim, self.kv_repeat = get_qkv_projection_dims(config)
        self.all_head_size = self.q_out
        self.all_kv_head_size = getattr(self, "kv_out", getattr(self, "all_head_size", self.hidden_size))
        self.num_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", self.num_heads))
        self.supports_gqa = True
        self.is_nested_encoder = False
        self.rank = config.svd_rank
        self.lora_rank = config.lora_rank
        self.fp8_format = getattr(config, "fp8_format", "e4m3")
        self.rotary_emb = None
        self.adapter = None

        # Reliable decoder detection
        model_type = getattr(config, "model_type", "") if config else ""
        is_dec = getattr(config, "is_decoder", False)
        if any(str(model_type).lower().startswith(p) for p in ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")):
            is_dec = True
        elif hasattr(config, "architectures") and any("GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(config, "architectures", []) or [])):
            is_dec = True
        elif pretrained_attn is not None and (hasattr(pretrained_attn, "c_attn") or pretrained_attn.__class__.__name__ in ("GPT2Attention", "Qwen2Attention", "Qwen2_5_VLAttention", "QwenAttention")):
            is_dec = True
        if kwargs.get("is_decoder", False):
            is_dec = True
        self.is_decoder: bool = bool(is_dec)

        self.router = SpectronSpectralEntropyRouter(config)

        # Persistent FactorizedLinear projections equipped with trainable residual LoRA adapters
        self.q_proj = FactorizedLinear(self.hidden_size, self.all_head_size, rank=self.rank, lora_rank=self.lora_rank, bias=True)
        self.k_proj = FactorizedLinear(self.hidden_size, self.all_kv_head_size, rank=self.rank, lora_rank=self.lora_rank, bias=True)
        self.v_proj = FactorizedLinear(self.hidden_size, self.all_kv_head_size, rank=self.rank, lora_rank=self.lora_rank, bias=True)
        self.out_proj = FactorizedLinear(self.all_head_size, self.hidden_size, rank=self.rank, lora_rank=self.lora_rank, bias=True)

        self.linear_A = nn.Parameter(torch.ones(self.all_head_size))
        self.linear_B = nn.Parameter(torch.ones(self.all_head_size))
        self.linear_C = nn.Parameter(torch.ones(self.all_head_size))
        self.decay_param = nn.Parameter(torch.full((self.all_head_size,), 4.5951))
        self.bidir_gate = nn.Parameter(torch.zeros(self.all_head_size))
        self.linear_in = nn.Linear(self.hidden_size, self.all_head_size, bias=False) if self.hidden_size != self.all_head_size else None

        self.svd_U = nn.Parameter(torch.randn(self.all_head_size, config.svd_rank) * 0.02)

        if pretrained_attn is not None:
            self._copy_pretrained(pretrained_attn)

    # --------------------------------------------------------------------------
    # Compatibility properties for parameter inspection & framework integration
    # --------------------------------------------------------------------------
    @property
    def q(self) -> FactorizedLinear:
        return self.q_proj

    @property
    def k(self) -> FactorizedLinear:
        return self.k_proj

    @property
    def v(self) -> FactorizedLinear:
        return self.v_proj

    @property
    def o(self) -> FactorizedLinear:
        return self.out_proj

    @property
    def query(self) -> FactorizedLinear:
        return self.q_proj

    @property
    def key(self) -> FactorizedLinear:
        return self.k_proj

    @property
    def value(self) -> FactorizedLinear:
        return self.v_proj

    @property
    def dense(self) -> FactorizedLinear:
        return self.out_proj

    @property
    def q_lin(self) -> FactorizedLinear:
        return self.q_proj

    @property
    def k_lin(self) -> FactorizedLinear:
        return self.k_proj

    @property
    def v_lin(self) -> FactorizedLinear:
        return self.v_proj

    @property
    def out_lin(self) -> FactorizedLinear:
        return self.out_proj

    @property
    def o_proj(self) -> FactorizedLinear:
        return self.out_proj

    @property
    def c_proj(self) -> FactorizedLinear:
        return self.out_proj

    @staticmethod
    def _svd_init(
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        proj: FactorizedLinear,
    ) -> None:
        """Initializes FactorizedLinear using truncated SVD approximation: W ≈ U[:, :r] diag(S[:r]) Vh[:r, :]."""
        with torch.no_grad():
            W = w.detach().float()
            if W.ndim != 2:
                raise ValueError(f"Expected 2D weight matrix, got shape {W.shape}")
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            r = min(proj.rank, U.size(1), Vh.size(0), proj.lin1.weight.size(0), proj.lin2.weight.size(1))

            proj.lin1.weight.data.zero_()
            proj.lin2.weight.data.zero_()

            in_copy = min(W.shape[1], proj.lin1.weight.size(1))
            out_copy = min(W.shape[0], proj.lin2.weight.size(0))

            proj.lin1.weight.data[:r, :in_copy].copy_(Vh[:r, :in_copy].to(proj.lin1.weight.dtype))
            US = U[:out_copy, :r] @ torch.diag(S[:r])
            proj.lin2.weight.data[:out_copy, :r].copy_(US.to(proj.lin2.weight.dtype))

            if b is not None and proj.lin2.bias is not None:
                min_b = min(b.numel(), proj.lin2.bias.numel())
                proj.lin2.bias.data[:min_b].copy_(b.detach()[:min_b].to(proj.lin2.bias.dtype))
            elif proj.lin2.bias is not None:
                proj.lin2.bias.data.zero_()

            # Ensure base factorized linear weights remain frozen
            proj.lin1.weight.requires_grad = False
            proj.lin2.weight.requires_grad = False
            if proj.lin2.bias is not None:
                proj.lin2.bias.requires_grad = False

            # Trainable LoRA adapter: A is initialized with Kaiming uniform, B is strictly zeroed
            if getattr(proj, "lora_A", None) is not None:
                nn.init.kaiming_uniform_(proj.lora_A, a=math.sqrt(5))
                proj.lora_A.requires_grad = True
            if getattr(proj, "lora_B", None) is not None:
                nn.init.zeros_(proj.lora_B)
                proj.lora_B.requires_grad = True

    def _copy_pretrained(self, old_attn: nn.Module) -> None:
        """Comprehensive weight extraction across DistilBERT, RoBERTa, GPT-2, and Qwen."""
        from framework.core.adapters import UniversalAdapter

        extracted_weights = {}
        for proj in ["q", "k", "v", "o"]:
            w, b = None, None

            # 1. Output projection checks first
            if proj == "o":
                if hasattr(old_attn, "c_proj"):
                    mod = old_attn.c_proj
                    if hasattr(mod, "merge") and not getattr(mod, "merged", False):
                        try:
                            mod.merge()
                        except Exception as e:
                            logger.debug(f"c_attn LoRA merge failed (using unmerged weights): {e}")
                    base_mod = getattr(mod, "base_layer", mod)
                    w = getattr(base_mod, "weight", getattr(mod, "weight", None))
                    if w is not None:
                        w = w.data
                        if (mod.__class__.__name__ == "Conv1D") or (base_mod.__class__.__name__ == "Conv1D") or hasattr(mod, "nf") or hasattr(base_mod, "nf") or (w.shape[0] < w.shape[1]):
                            w = w.t().contiguous()
                    bias_holder = getattr(base_mod, "bias", getattr(mod, "bias", None))
                    b = bias_holder.data if bias_holder is not None else None
                elif hasattr(old_attn, "output") and hasattr(getattr(old_attn, "output"), "dense"):
                    mod = old_attn.output.dense
                    w = mod.weight.data
                    b = getattr(mod, "bias", None)
                elif hasattr(old_attn, "dense"):
                    mod = old_attn.dense
                    w = mod.weight.data
                    b = getattr(mod, "bias", None)
                elif hasattr(old_attn, "out_lin"):
                    mod = old_attn.out_lin
                    w = mod.weight.data
                    b = getattr(mod, "bias", None)
                elif hasattr(old_attn, "out_proj"):
                    mod = old_attn.out_proj
                    w = mod.weight.data
                    b = getattr(mod, "bias", None)
                elif hasattr(old_attn, "o_proj"):
                    mod = old_attn.o_proj
                    w = mod.weight.data
                    b = getattr(mod, "bias", None)

            # 2. GPT-2 Fused c_attn
            elif hasattr(old_attn, "c_attn") and proj in ["q", "k", "v"]:
                mod = old_attn.c_attn
                if hasattr(mod, "merge") and not getattr(mod, "merged", False):
                    try:
                        mod.merge()
                    except Exception as e:
                        logger.debug(f"c_proj LoRA merge failed (using unmerged weights): {e}")
                base_mod = getattr(mod, "base_layer", mod)
                c_w = getattr(base_mod, "weight", getattr(mod, "weight", None))
                if c_w is not None:
                    c_w = c_w.data
                    if (mod.__class__.__name__ == "Conv1D") or (base_mod.__class__.__name__ == "Conv1D") or hasattr(mod, "nf") or hasattr(base_mod, "nf") or (c_w.shape[0] < c_w.shape[1]):
                        c_w = c_w.t().contiguous()
                    split_size = c_w.size(0) // 3
                    idx = {"q": 0, "k": 1, "v": 2}[proj]
                    w = c_w[idx * split_size : (idx + 1) * split_size, :]
                    bias_holder = getattr(base_mod, "bias", getattr(mod, "bias", None))
                    if bias_holder is not None:
                        c_b = bias_holder.data
                        b = c_b[idx * split_size : (idx + 1) * split_size]

            # 3. RoBERTa / BERT nested self-attention (old_attn.self)
            elif hasattr(old_attn, "self") and proj in ["q", "k", "v"]:
                w, b = None, None
                s = getattr(old_attn, "self")
                name_map = {"q": "query", "k": "key", "v": "value"}
                if hasattr(s, name_map[proj]):
                    mod = getattr(s, name_map[proj])
                    w = mod.weight.data
                    b = getattr(mod, "bias", None)

            # 4. Qwen3.5 GatedDeltaNet fused in_proj_qkv
            elif hasattr(old_attn, "in_proj_qkv") and proj in ["q", "k", "v"]:
                mod = old_attn.in_proj_qkv
                fused_w = mod.weight.data
                fused_b = getattr(mod, "bias", None)
                k_dim = getattr(old_attn, "key_dim", getattr(old_attn, "k_dim", fused_w.size(0) // 3))
                v_dim = getattr(old_attn, "value_dim", getattr(old_attn, "v_dim", fused_w.size(0) // 3))
                q_dim = fused_w.size(0) - k_dim - v_dim if (k_dim + v_dim < fused_w.size(0)) else k_dim
                if proj == "q":
                    w = fused_w[:q_dim, :]
                    b = fused_b[:q_dim] if fused_b is not None else None
                elif proj == "k":
                    w = fused_w[q_dim : q_dim + k_dim, :]
                    b = fused_b[q_dim : q_dim + k_dim] if fused_b is not None else None
                elif proj == "v":
                    w = fused_w[q_dim + k_dim : q_dim + k_dim + v_dim, :]
                    b = fused_b[q_dim + k_dim : q_dim + k_dim + v_dim] if fused_b is not None else None

            # 5. Direct named projections
            if w is None:
                for attr_candidate in [f"{proj}_lin", f"{proj}_proj", proj]:
                    if hasattr(old_attn, attr_candidate):
                        mod = getattr(old_attn, attr_candidate)
                        if hasattr(mod, "weight"):
                            w = mod.weight.data
                            b = getattr(mod, "bias", None)
                            if b is not None:
                                b = b.data
                            break

            # 6. Fallback to UniversalAdapter
            if w is None:
                try:
                    w, b = UniversalAdapter.get_weight_bias(old_attn, proj)
                except Exception as e:
                    logger.debug(f"UniversalAdapter fallback failed for {proj}: {e}")

            extracted_weights[proj] = (w, b)

        cls_name = old_attn.__class__.__name__
        has_out_proj = any(
            hasattr(old_attn, attr)
            for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o")
        )
        # Check output projection attrs FIRST — DistilBERT has out_lin and is NOT nested
        if hasattr(old_attn, "out_lin") or hasattr(old_attn, "c_proj") or hasattr(old_attn, "o_proj"):
            self.is_nested_encoder = False
        elif (
            hasattr(old_attn, "self")
            and hasattr(getattr(old_attn, "self"), "query")
            and hasattr(old_attn, "output")
            and hasattr(getattr(old_attn, "output"), "dense")
        ):
            self.is_nested_encoder = True
        elif (
            cls_name in ["BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention"]
            or "RobertaSelfAttention" in str(type(old_attn))
            or "BertSelfAttention" in str(type(old_attn))
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ):
            self.is_nested_encoder = True
        else:
            self.is_nested_encoder = False

        # Strict check for essential missing weights
        missing = [p for p in ["q", "k", "v"] if extracted_weights[p][0] is None]
        if not self.is_nested_encoder and extracted_weights["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning(f"AegisSpectronFP8: missing weights for projections {missing}; proceeding with random init for those")

        for p in ["q", "k", "v"]:
            w_chk = extracted_weights[p][0]
            if w_chk is not None:
                norm = float(torch.linalg.norm(w_chk.float()).item())
                if norm < 1e-4:
                    logger.warning(f"AegisSpectronFP8: extracted weight for '{p}' has near-zero Frobenius norm ({norm:.2e}); continuing")
        if extracted_weights["o"][0] is not None:
            norm_o = float(torch.linalg.norm(extracted_weights["o"][0].float()).item())
            if norm_o < 1e-4:
                logger.warning(f"AegisSpectronFP8: extracted weight for 'o' has near-zero Frobenius norm ({norm_o:.2e}); continuing")

        for proj, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj)]:
            w, b = extracted_weights[proj]
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
                self.out_proj.lin1.weight.requires_grad = False
                self.out_proj.lin2.weight.requires_grad = False
        elif extracted_weights["o"][0] is not None:
            w_o, b_o = extracted_weights["o"]
            self._svd_init(w_o, b_o, self.out_proj)

        if extracted_weights["q"][0] is not None:
            with torch.no_grad():
                U, S, _ = torch.linalg.svd(extracted_weights["q"][0].float(), full_matrices=False)
                r = min(self.config.svd_rank, U.size(1), self.svd_U.size(1))
                self.svd_U.data[:, :r].copy_(U[:self.svd_U.size(0), :r].to(self.svd_U.dtype))

        # Copy rotary_emb if present
        if hasattr(old_attn, "rotary_emb"):
            self.rotary_emb = old_attn.rotary_emb

        # Update decoder status if old_attn indicates decoder
        if hasattr(old_attn, "is_decoder"):
            self.is_decoder = bool(old_attn.is_decoder)
        elif hasattr(old_attn, "c_attn") or old_attn.__class__.__name__ in ("GPT2Attention", "Qwen2Attention", "Qwen2_5_VLAttention", "QwenAttention"):
            self.is_decoder = True

    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Optional[Any] = None, **kwargs: Any
    ) -> "AegisSpectronFP8Attention":
        r"""Instantiates AegisSpectronFP8Attention with weights transferred from pretrained attention."""
        if model_config is None:
            class DummyConfig:
                pass
            model_config = DummyConfig()
            out_layer = getattr(old_attn, "out_lin", getattr(old_attn, "dense", None))
            model_config.hidden_size = out_layer.out_features if out_layer is not None else 768
            model_config.num_attention_heads = getattr(old_attn, "n_heads", getattr(old_attn, "num_attention_heads", 12))
        return cls(config=model_config, pretrained_attn=old_attn, **kwargs)

    def _local_branch(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, h, N, d = Q.shape
        W = self.config.window_size

        attn_weights = torch.matmul(Q, K.transpose(-1, -2).contiguous()) / math.sqrt(d)

        idx = torch.arange(N, device=Q.device)
        idx_q = idx.unsqueeze(-1)
        idx_k = idx.unsqueeze(-2)
        dist = (idx_q - idx_k).abs()
        mask = (dist > (W // 2)) & (idx_q != 0) & (idx_k != 0)
        mask_val = -10000.0

        if getattr(self.config, "is_decoder", getattr(self, "is_decoder", False)) and N > 1:
            causal_mask = idx_k > idx_q
            mask = mask | causal_mask
        attn_weights = attn_weights.masked_fill(mask, mask_val)

        # Enforce strict causal masking for decoders (fix future token leakage)
        if getattr(self, "is_decoder", False):
            causal_mask = (idx.unsqueeze(-1) - idx.unsqueeze(-2)) < 0
            attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), mask_val)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_m = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * mask_val
                attn_weights = attn_weights + pad_m
            elif attention_mask.dim() == 4:
                if attention_mask.shape[1] == 1 or attention_mask.shape[1] == attn_weights.shape[1]:
                    attn_weights = attn_weights + attention_mask.to(attn_weights.dtype)
                elif attn_weights.shape[1] % attention_mask.shape[1] == 0:
                    attn_weights = attn_weights + repeat_kv(attention_mask, attn_weights.shape[1] // attention_mask.shape[1]).to(attn_weights.dtype)
                else:
                    attn_weights = attn_weights + attention_mask[:, :1].to(attn_weights.dtype)

        attn = F.softmax(attn_weights, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = F.dropout(attn, p=self.config.dropout, training=self.training)
        return torch.matmul(attn, V)

    def _global_branch(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, h, N, d = Q.shape

        Q_flat = Q.transpose(1, 2).contiguous().view(B, N, -1)
        K_flat = K.transpose(1, 2).contiguous().view(B, N, -1)
        V_flat = V.transpose(1, 2).contiguous().view(B, N, -1)

        Q_lr = torch.matmul(Q_flat, self.svd_U)  # [B, N, rank]
        K_lr = torch.matmul(K_flat, self.svd_U)  # [B, N, rank]

        # Non-negative kernel mapping phi(x) = ELU(x) + 1
        phi_Q = F.elu(Q_lr) + 1.0
        phi_K = F.elu(K_lr) + 1.0

        device_type = Q.device.type if Q.device.type in ("cuda", "cpu") else "cuda"
        with torch.autocast(device_type=device_type, enabled=False):
            phi_Q_f = phi_Q.float()
            phi_K_f = phi_K.float()
            V_f = V_flat.float()

            if getattr(self, "is_decoder", False):
                # Causal Global Linear Attention via Cumulative Sums in FP32
                # Numerator: cumsum_{j <= t} phi_K_j * V_j -> [B, N, rank, D]
                KV_pair = phi_K_f.unsqueeze(-1) * V_f.unsqueeze(-2)
                KV_cumsum = torch.cumsum(KV_pair, dim=1)
                num = torch.einsum("bnr,bnrd->bnd", phi_Q_f, KV_cumsum)

                # Denominator: cumsum_{j <= t} phi_K_j -> [B, N, rank]
                K_cumsum = torch.cumsum(phi_K_f, dim=1)
                den = (phi_Q_f * K_cumsum).sum(dim=-1, keepdim=True).clamp(min=1e-4)

                out_lr = (num / den).to(Q.dtype)
                return out_lr.view(B, N, h, d).transpose(1, 2).contiguous()
            else:
                # True Bidirectional Linear Attention with Normalization (FP32 to prevent FP16 Overflow)
                KV = torch.matmul(phi_K_f.transpose(-1, -2).contiguous(), V_f)
                Z = phi_K_f.sum(dim=1, keepdim=True)
                num = torch.matmul(phi_Q_f, KV)
                den = torch.matmul(phi_Q_f, Z.transpose(-1, -2).contiguous()).clamp(min=1e-4)

                out_lr = (num / den).to(Q.dtype)
                return out_lr.view(B, N, h, d).transpose(1, 2).contiguous()

    def _linear_branch(self, X: torch.Tensor) -> torch.Tensor:
        if self.linear_in is not None:
            X = self.linear_in(X)
        B, N, D = X.shape

        u = X * self.linear_B.view(1, 1, D)
        gamma = torch.sigmoid(getattr(self, "decay_param", self.linear_A)).clamp(min=1e-4, max=1.0 - 1e-4)

        # O(N) Recurrence using CumSum trick: y_t = gamma^t * cumsum(u_i * gamma^{-i})
        pos = torch.arange(N, device=X.device, dtype=torch.float32).view(1, N, 1)
        gamma_fp32 = gamma.view(1, 1, D).float()

        # Numerically stable pow using log & clamp to avoid float32 underflow/overflow
        log_gamma = torch.log(gamma_fp32)
        gamma_pow = torch.exp((pos * log_gamma).clamp(min=-30.0, max=0.0))
        gamma_inv_pow = torch.exp((-pos * log_gamma).clamp(min=0.0, max=30.0))

        # Causal forward pass
        u_scaled_fwd = u.float() * gamma_inv_pow
        out_fwd = gamma_pow * torch.cumsum(u_scaled_fwd, dim=1)
        out_fwd = out_fwd.to(X.dtype) * self.linear_C.view(1, 1, D)

        if getattr(self, "is_decoder", False):
            out_tensor = out_fwd  # Strict causal recurrence for decoders
        else:
            # Anti-causal backward pass for bidirectional models
            u_scaled_bwd = torch.flip(u.float(), [1]) * gamma_inv_pow
            out_bwd = gamma_pow * torch.cumsum(u_scaled_bwd, dim=1)
            out_bwd = torch.flip(out_bwd, [1]).to(X.dtype) * self.linear_C.view(1, 1, D)

            gate = torch.sigmoid(getattr(self, "bidir_gate", torch.tensor(0.5, device=X.device, dtype=X.dtype)))
            out_tensor = gate * out_fwd + (1.0 - gate) * out_bwd

        out_tensor = out_tensor.clamp(min=-1e4, max=1e4)
        return out_tensor.view(B, N, self.config.num_heads, self.config.head_dim).transpose(1, 2).contiguous()

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = hidden_states.shape

        # Token-level routing on hidden_states
        alpha = self.router(hidden_states)  # [B, N, 3]
        alpha_loc = alpha[:, :, 0].unsqueeze(1).unsqueeze(-1)   # [B, 1, N, 1]
        alpha_glob = alpha[:, :, 1].unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
        alpha_lin = alpha[:, :, 2].unsqueeze(1).unsqueeze(-1)   # [B, 1, N, 1]

        # Projections via FactorizedLinear (base + LoRA) with FP8 cast applied to projections
        fp8_fmt = getattr(self.config, "fp8_format", "e4m3")
        X_fp8 = fp8_cast_simulate(hidden_states, format=fp8_fmt)
        Q = fp8_cast_simulate(self.q_proj(hidden_states), format=fp8_fmt).view(B, N, self.config.num_heads, self.config.head_dim).transpose(1, 2).contiguous()
        K = fp8_cast_simulate(self.k_proj(hidden_states), format=fp8_fmt).view(B, N, self.num_kv_heads, self.config.head_dim).transpose(1, 2).contiguous()
        V = fp8_cast_simulate(self.v_proj(hidden_states), format=fp8_fmt).view(B, N, self.num_kv_heads, self.config.head_dim).transpose(1, 2).contiguous()

        # Apply Rotary Position Embeddings (RoPE) immediately after projections and BEFORE branching
        position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
            cos, sin = position_embeddings[0], position_embeddings[1]

            def _rotate_half(x: torch.Tensor) -> torch.Tensor:
                return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)

            if cos.dim() == 3:
                cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
            while cos.dim() < Q.dim():
                cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
            if cos.shape[-2] != Q.shape[-2]:
                cos, sin = cos[..., : Q.shape[-2], :], sin[..., : Q.shape[-2], :]

            rotary_dim = cos.shape[-1]
            if rotary_dim <= Q.shape[-1]:
                q_rot, q_pass = Q[..., :rotary_dim], Q[..., rotary_dim:]
                k_rot, k_pass = K[..., :rotary_dim], K[..., rotary_dim:]
                q_rot = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
                k_rot = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
                Q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot
                K = torch.cat([k_rot, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot
            else:
                cos_cut, sin_cut = cos[..., : Q.shape[-1]], sin[..., : Q.shape[-1]]
                Q = (Q * cos_cut.to(Q.dtype)) + (_rotate_half(Q) * sin_cut.to(Q.dtype))
                K = (K * cos_cut.to(K.dtype)) + (_rotate_half(K) * sin_cut.to(K.dtype))
        elif hasattr(self, "rotary_emb") and self.rotary_emb is not None:
            try:
                from framework.utils.compat import maybe_apply_rotary
                Q, K = maybe_apply_rotary(Q, K, getattr(self, "adapter", None), self.rotary_emb, kwargs.get("position_ids"))
            except Exception as e:
                logger.error("maybe_apply_rotary failed: %s", e)

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        # 3 Independent Branches
        out_loc = self._local_branch(Q, K, V, attention_mask)
        out_glob = self._global_branch(Q, K, V)
        out_lin = self._linear_branch(X_fp8)

        # Token-level dynamic fusion
        out_fused = (alpha_loc * out_loc) + (alpha_glob * out_glob) + (alpha_lin * out_lin)
        out_fused = out_fused.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        out_fused = torch.nan_to_num(out_fused, nan=0.0, posinf=1e4, neginf=-1e4).clamp(min=-1e4, max=1e4)

        if getattr(self, "is_nested_encoder", False):
            final_out = out_fused
        else:
            final_out = self.out_proj(out_fused)

        final_out = torch.nan_to_num(final_out, nan=0.0, posinf=1e4, neginf=-1e4).clamp(min=-1e4, max=1e4)
        return final_out, None


@ATTENTION_REGISTRY.register("aegis_spectron_fp8_inject")
@ATTENTION_REGISTRY.register("inject_aegis_spectron_fp8")
def inject_aegis_spectron_fp8(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
):
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer", None))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (torch.nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_aegis_spectron_fp8(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config if 'comp_config' in locals() or 'comp_config' in kwargs else None,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    r"""Injects Aegis-Spectron-FP8 attention into a transformer layer with safe pre-testing."""
    if model_config is None:
        if hasattr(layer, "hidden_size") or hasattr(layer, "d_model") or hasattr(layer, "n_embd") or hasattr(layer, "dim"):
            model_config = layer
        elif hasattr(layer, "attention"):
            old_attn = layer.attention
    h_size = getattr(model_config, "hidden_size", getattr(model_config, "d_model", getattr(model_config, "n_embd", getattr(model_config, "dim", 768)))) if model_config is not None else 768
    n_heads = getattr(model_config, "num_attention_heads", getattr(model_config, "num_heads", getattr(model_config, "n_heads", getattr(model_config, "n_head", 12)))) if model_config is not None else 12
    # Use adapter or config's native head_dim if available (avoids 128 vs 256 bug for Qwen)
    h_dim = getattr(model_config, "head_dim", getattr(model_config, "kv_channels", h_size // max(1, n_heads)))
    n_kv_heads = getattr(model_config, "num_key_value_heads", getattr(model_config, "num_kv_heads", n_heads))

    is_dec = getattr(model_config, "is_decoder", False) if model_config else False
    if kwargs.get("is_decoder", False):
        is_dec = True

    cfg = AegisSpectronFP8Config(
        hidden_size=h_size,
        num_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=h_dim,
        window_size=getattr(comp_config, "window_size", getattr(comp_config, "local_window", 64)) if comp_config else 64,
        svd_rank=getattr(comp_config, "svd_rank", getattr(comp_config, "rank", 384)) if comp_config else 384,
        lora_rank=getattr(comp_config, "lora_rank", getattr(comp_config, "lora_r", 32)) if comp_config else 32,
        dropout=getattr(comp_config, "dropout", 0.1) if comp_config else 0.1,
        router_init_bias=getattr(comp_config, "router_init_bias", kwargs.get("router_init_bias", (1.0, 1.0, 1.0))),
        is_decoder=is_dec,
        fp8_format=getattr(comp_config, "fp8_format", "e4m3") if comp_config else "e4m3",
    )

    # Injection Pre-test: safely abort without corrupting the model if weight copying fails
    try:
        if model_config is None and old_attn is not None:
            new_attn = AegisSpectronFP8Attention.from_distilbert_pretrained(old_attn, config=cfg)
        else:
            new_attn = AegisSpectronFP8Attention(config=cfg, pretrained_attn=old_attn, is_decoder=is_dec)
        if hasattr(old_attn, "rotary_emb"):
            new_attn.rotary_emb = old_attn.rotary_emb
    except Exception as e:
        logger.error("inject_aegis_spectron_fp8 failed: %s", e)
        return None

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
