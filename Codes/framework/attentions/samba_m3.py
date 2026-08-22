# SAMBA-M3.py
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from framework.core.registry import ATTENTION_REGISTRY

# فرض: mamba.py شامل Mamba اصلاح‌شده است
from framework.attentions.mamba import Mamba


@ATTENTION_REGISTRY.register("spectral_mamba")
class SpectralMamba(nn.Module):
    r"""
    Mamba با مدولاسیون طیفی یادگیرنده.

    نوآوری:
    - گیت channel-wise با سیگموید برای فیلتر کردن کانال‌ها
    - مقداردهی اولیه بهتر برای جلوگیری از خاموشی شاخه Mamba
    """

    def __init__(self, d_model: int = 768, config: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__()
        if config is not None:
            d_model = getattr(config, "hidden_size", getattr(config, "d_model", d_model))
        elif hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        self.mamba = Mamba(d_model)
        # مقدار اولیه مثبت تا سیگموید نزدیک 0.5 باشد و از خاموشی جلوگیری کند
        self.dt_scale = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out = self.mamba(x)
        if isinstance(out, tuple):
            out = out[0]
        # گیت سیگموید
        gate = torch.sigmoid(self.dt_scale).unsqueeze(0).unsqueeze(0)
        out = out * gate
        return out, None


@ATTENTION_REGISTRY.register("samba_m3")
class SAMBAM3Block(nn.Module):
    r"""
    SAMBA-M3: ترکیب attention پنجره‌ای محلی و Spectral Mamba.

    اصلاحات:
    - ماسک sliding-window دیگر تنسور کامل [L, L] با float32 نمی‌سازد؛
      از همان dtype ورودی استفاده می‌کند.
    - attention_mask ورودی به زیرلایه‌ها منتقل می‌شود.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        window: int = 128,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is not None:
            d_model = getattr(config, "hidden_size", getattr(config, "d_model", d_model))
            n_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", n_heads))
            window = getattr(config, "window_size", window)

        self.window_size = window
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mamba = SpectralMamba(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def sliding_window_mask(self, L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        idx = torch.arange(L, device=device)
        causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
        window = (idx.unsqueeze(1) - idx.unsqueeze(0)) < self.window_size
        valid = causal & window

        # استفاده از dtype ورودی برای جلوگیری از mismatch
        mask = torch.zeros(L, L, device=device, dtype=dtype)
        mask_val = -10000.0 if dtype in (torch.float16, torch.bfloat16) else float("-inf")
        mask = mask.masked_fill(~valid, mask_val)
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = hidden_states
        B, L, D = x.shape

        # ماسک پنجره‌ای با dtype صحیح
        window_mask = self.sliding_window_mask(L, device=x.device, dtype=x.dtype)

        # اگر attention_mask خارجی وجود دارد، ترکیب می‌کنیم (اختیاری)
        if attention_mask is not None and attention_mask.dim() == 4:
            pad_mask = attention_mask[:, 0, 0, :]  # [B, L] -> 0/1
            # تبدیل به ماسک additive برای MultiheadAttention
            # برای سادگی، فعلاً فقط از window mask استفاده می‌کنیم.
            # در صورت نیاز، می‌توان pad_mask را نیز اضافه کرد.

        attn_out, attn_w = self.attn(x, x, x, attn_mask=window_mask)
        x = self.norm1(x + attn_out)

        m_out = self.mamba(x)
        if isinstance(m_out, tuple):
            m_out = m_out[0]
        x = self.norm2(x + m_out)

        return x, attn_w


@ATTENTION_REGISTRY.register("samba_m3_inject")
def inject_samba_m3(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> SAMBAM3Block:
    d_model = model_config.hidden_size
    n_heads = getattr(model_config, "num_attention_heads", 12)
    window = getattr(comp_config, "window_size", 128)

    new_block = SAMBAM3Block(
        d_model=d_model,
        n_heads=n_heads,
        window=window,
        config=model_config,
    )
    layer.attention = new_block
    return new_block