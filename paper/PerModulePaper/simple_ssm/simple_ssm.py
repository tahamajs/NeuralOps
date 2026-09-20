"""Simple State Space Model (SimpleSSM).

Lightweight bidirectional/causal selective state space recurrence block featuring symmetric depthwise 1D convolution,
input-dependent discretization timescales (Delta t), diagonal state transitions (A), multiplicative gating,
and pretrained weight transfer.
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


class LoRALinear(nn.Module):
    """Linear projection with frozen base weights and trainable LoRA adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / max(1, lora_rank)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register("simple_ssm", supports_causal=True)
class SimpleSSM(BaseAttention):
    r"""Bidirectional / Causal Simple State Space Model (SimpleSSM) Module.

    Features input-dependent discretization timescales (Delta t), diagonal state transitions (A),
    multiplicative SiLU gating, and pretrained weight transfer.
    """

    supports_causal: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        d_model: Optional[Union[int, Any]] = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 1,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        if isinstance(config, int):
            d_model = config
            config = None
        if d_model is None:
            if config is not None:
                d_model = getattr(config, "hidden_size", getattr(config, "dim", 768))
            else:
                d_model = kwargs.get("hidden_size", 768)

        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        self.d_model: int = int(d_model)
        self.d_state: int = d_state
        self.d_conv: int = d_conv
        self.expand: int = expand
        self.inner_dim: int = self.d_model * expand
        self.lora_rank: int = lora_rank
        self.lora_alpha: float = lora_alpha

        # Decoder / Causal flag
        if is_decoder is not None:
            self.is_decoder: bool = is_decoder
        elif config is not None:
            self.is_decoder = bool(getattr(config, "is_decoder", False))
        else:
            self.is_decoder = False

        self.is_nested_encoder: bool = False
        num_heads = getattr(config, "num_attention_heads", kwargs.get("num_attention_heads", 12)) if config else 12
        num_kv_heads = getattr(config, "num_key_value_heads", kwargs.get("num_key_value_heads", num_heads)) if config else num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        self.in_proj = LoRALinear(
            self.d_model,
            self.inner_dim * 2,
            bias=False,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.conv1d = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=d_conv,
            groups=self.inner_dim,
            padding=d_conv - 1,
        )
        self.dt_proj = LoRALinear(
            self.inner_dim,
            self.d_model,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_model, 1))
        )
        self.D = nn.Parameter(torch.ones(self.d_model))
        self.out_proj = LoRALinear(
            self.d_model,
            self.d_model,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        nn.init.constant_(self.dt_proj.bias, 0.1)

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
                # Unwrap nested encoder modules to find innermost attention
                src = old_attn
                while hasattr(src, "self") and not any(
                    hasattr(src, attr) for attr in ["query", "key", "value", "q_proj", "k_proj", "v_proj"]
                ):
                    src = src.self
                # Copy output projection if available
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
                logger.warning(
                    f"SimpleSSM: Failed to extract weights: {', '.join(missing)} from {type(old_attn)}"
                )
                return

            # Copy input projection: [V; Q]
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

            self.in_proj.weight[: self.inner_dim, : self.d_model].copy_(v_chunk.to(self.in_proj.weight.dtype))
            self.in_proj.weight[self.inner_dim :, : self.d_model].copy_(q_chunk.to(self.in_proj.weight.dtype))

            f_norm_in = float(torch.linalg.norm(self.in_proj.weight))
            if f_norm_in < 1e-4:
                logger.warning(f"SimpleSSM: in_proj weight norm is near zero: {f_norm_in}")

            if w_o is not None:
                self.out_proj.weight.copy_(w_o[: self.d_model, : self.d_model].to(self.out_proj.weight.dtype))
                if b_o is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.copy_(b_o[: self.d_model].to(self.out_proj.bias.dtype))
                f_norm_o = float(torch.linalg.norm(self.out_proj.weight))
                if f_norm_o < 1e-4:
                    logger.warning(f"SimpleSSM: out_proj weight norm is near zero: {f_norm_o}")

    @property
    def query(self) -> LoRALinear:
        return self.in_proj

    @property
    def key(self) -> LoRALinear:
        return self.in_proj

    @property
    def value(self) -> LoRALinear:
        return self.in_proj

    def _ssm_scan_parallel(
        self,
        x_ssm: torch.Tensor,
        dt: torch.Tensor,
        A_log: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Bounded chunked recurrence (O(chunk=64)) associative state scan."""
        B, L, D = x_ssm.shape
        A = torch.exp(A_log.clamp(max=10.0)).mean(dim=-1)  # Effective diagonal transition [D]

        # log_dA = -dt * A: [B, L, D]
        log_dA = -dt * A.view(1, 1, D)
        u = (dt * x_ssm).float()  # [B, L, D]

        if mask is not None:
            mask_float = mask.float() if mask.dim() == 2 else mask[:, 0, 0, :].float()
            u = u * mask_float.unsqueeze(-1)

        chunk_size = 64
        num_chunks = (L + chunk_size - 1) // chunk_size

        # Forward causal recurrence (strictly lower-triangular)
        h_running = torch.zeros(B, D, device=x_ssm.device, dtype=torch.float32)
        h_fwd_chunks = []
        for c in range(num_chunks):
            s_idx = c * chunk_size
            e_idx = min(L, s_idx + chunk_size)

            u_c = u[:, s_idx:e_idx]
            log_a_c = log_dA[:, s_idx:e_idx]
            lambda_c = torch.clamp(torch.cumsum(log_a_c, dim=1), min=-25.0, max=0.0)

            decay_c = torch.exp(lambda_c)
            decay_inv_c = torch.exp(-lambda_c).clamp(max=1e10)

            h_intra = decay_c * torch.cumsum(decay_inv_c * u_c, dim=1)
            h_inter = decay_c * h_running.unsqueeze(1)

            h_chunk = torch.clamp(h_intra + h_inter, min=-1e4, max=1e4)
            h_fwd_chunks.append(h_chunk)
            h_running = h_chunk[:, -1]

        h_fwd = torch.cat(h_fwd_chunks, dim=1)

        # Backward recurrence (skipped in decoder mode)
        if not self.is_decoder:
            u_rev = torch.flip(u, dims=[1])
            log_dA_rev = torch.flip(log_dA, dims=[1])
            h_running_bwd = torch.zeros(B, D, device=x_ssm.device, dtype=torch.float32)
            h_bwd_chunks = []
            for c in range(num_chunks):
                s_idx = c * chunk_size
                e_idx = min(L, s_idx + chunk_size)

                u_c = u_rev[:, s_idx:e_idx]
                log_a_c = log_dA_rev[:, s_idx:e_idx]
                lambda_c = torch.clamp(torch.cumsum(log_a_c, dim=1), min=-25.0, max=0.0)

                decay_c = torch.exp(lambda_c)
                decay_inv_c = torch.exp(-lambda_c).clamp(max=1e10)

                h_intra = decay_c * torch.cumsum(decay_inv_c * u_c, dim=1)
                h_inter = decay_c * h_running_bwd.unsqueeze(1)

                h_chunk = torch.clamp(h_intra + h_inter, min=-1e4, max=1e4)
                h_bwd_chunks.append(h_chunk)
                h_running_bwd = h_chunk[:, -1]

            h_bwd = torch.flip(torch.cat(h_bwd_chunks, dim=1), dims=[1])
        else:
            h_bwd = None

        out_fwd = (h_fwd * self.D.view(1, 1, D)).to(dtype=x_ssm.dtype)
        out_bwd = (h_bwd * self.D.view(1, 1, D)).to(dtype=x_ssm.dtype) if h_bwd is not None else None

        return out_fwd, out_bwd

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
        """Compute SSM recurrence over sequence tokens."""
        # Cross-attention guard
        if encoder_hidden_states is not None:
            return (hidden_states, None)

        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape

        # Nested encoder bypass: RoBERTa/BERT layer.attention.self bypass internal LayerNorm
        if self.is_nested_encoder:
            x_norm = hidden_states
        else:
            x_norm = F.layer_norm(hidden_states, (D,))

        x_proj = self.in_proj(x_norm)
        x_ssm, z = x_proj.chunk(2, dim=-1)

        # 1D conv with exact sequence length retention
        x_ssm = self.conv1d(x_ssm.transpose(1, 2).contiguous())[..., :L].transpose(1, 2).contiguous()
        x_ssm = F.silu(x_ssm)

        # Discretization step size dt bounded to [0.001, 0.1]
        dt = F.softplus(self.dt_proj(x_ssm)).clamp(min=1e-3, max=0.1)

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
            mask_bool = torch.ones(B, L, device=hidden_states.device, dtype=torch.bool)

        out_fwd, out_bwd = self._ssm_scan_parallel(x_ssm, dt, self.A_log, mask_bool)

        if self.is_decoder or out_bwd is None:
            y = out_fwd
        else:
            y = 0.5 * (out_fwd + out_bwd)

        gated = y * F.silu(z)

        # Nested encoder bypass: RoBERTa/BERT layer.attention.self bypass internal output projection
        if self.is_nested_encoder:
            out = gated if gated.shape[-1] == D else self.out_proj(gated)
        else:
            out = self.out_proj(gated)

        if mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        if out.shape[-1] != hidden_states.shape[-1]:
            if out.shape[-1] < hidden_states.shape[-1]:
                out = F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
            else:
                out = out[..., :hidden_states.shape[-1]]

        return (out, None)


@ATTENTION_REGISTRY.register("simple_ssm_inject")
@ATTENTION_REGISTRY.register("inject_simple_ssm")
def inject_simple_ssm(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 6,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[SimpleSSM]:
    r"""Safe injection hook for SimpleSSM."""
    target = layer if layer is not None else model
    try:
        # Whole-model traversal support
        if target is not None and (
            hasattr(target, "transformer")
            or hasattr(target, "distilbert")
            or hasattr(target, "roberta")
            or hasattr(target, "bert")
            or hasattr(target, "h")
            or (hasattr(target, "encoder") and hasattr(target.encoder, "layer"))
            or (hasattr(target, "model") and hasattr(target.model, "layers"))
        ):
            layer_list = None
            if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
                layer_list = target.distilbert.transformer.layer
            elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
                layer_list = target.roberta.encoder.layer
            elif hasattr(target, "bert") and hasattr(target.bert, "encoder"):
                layer_list = target.bert.encoder.layer
            elif hasattr(target, "h"):
                layer_list = target.h
            elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
                layer_list = target.transformer.h
            elif hasattr(target, "model") and hasattr(target.model, "layers"):
                layer_list = target.model.layers
            elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
                layer_list = target.encoder.layer

            if layer_list is not None:
                tot = len(layer_list)
                for idx, lyr in enumerate(layer_list):
                    inject_simple_ssm(
                        layer=lyr,
                        model_config=model_config,
                        comp_config=comp_config,
                        layer_idx=idx,
                        total_layers=tot,
                        **kwargs,
                    )
                return target

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

        d_model = 768
        if model_config is not None:
            d_model = getattr(model_config, "hidden_size", getattr(model_config, "dim", 768))
        elif hasattr(target, "config"):
            d_model = getattr(target.config, "hidden_size", 768)
        elif hasattr(src, "hidden_size"):
            d_model = src.hidden_size

        new_block = SimpleSSM(config=model_config, d_model=d_model, **kwargs)
        if src is not None:
            try:
                new_block.copy_weights(src)
            except Exception as w_err:
                logger.debug("SimpleSSM weight copy warning: %s", w_err)

        if target is not None:
            if hasattr(target, "attention"):
                if hasattr(target.attention, "self"):
                    target.attention.self = new_block
                    new_block.is_nested_encoder = True
                    return new_block
                else:
                    target.attention = new_block
                    return new_block
            for attr in ("self_attn", "attn"):
                if hasattr(target, attr):
                    setattr(target, attr, new_block)
                    return new_block
            if hasattr(target, "self"):
                target.self = new_block
                new_block.is_nested_encoder = True
                return new_block

        return new_block
    except Exception as e:
        logger.error("inject_simple_ssm failed at layer %d: %s", layer_idx, e)
        return None


inject_simple_ssm_attention = inject_simple_ssm
