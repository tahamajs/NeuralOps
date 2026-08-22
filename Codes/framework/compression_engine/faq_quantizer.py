"""FAQ (Feature-Activation-Aware Quantization): adjusts quantization scales using activation statistics. Registered as 'faq'."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from framework.compression_engine.astro_quantizer import (
    STEFakeQuantize,
    quantize_dequantize_tensor,
)
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class FAQLinear(nn.Module):
    """Linear layer using FAQ activation-aware quantization with STE backpropagation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 8,
        symmetric: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.symmetric = symmetric

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.register_buffer("act_scale", torch.ones(in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        act_scale: torch.Tensor,
        bits: int = 8,
        symmetric: bool = True,
    ) -> FAQLinear:
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            bits=bits,
            symmetric=symmetric,
        )
        layer.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        layer.act_scale.copy_(act_scale.view(-1)[: linear.in_features])
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale weights by activation sensitivity for quantization
        s = torch.clamp(self.act_scale, min=1e-6).unsqueeze(0)
        w_scaled = self.weight * s
        w_q = quantize_dequantize_tensor(
            w_scaled,
            bits=self.bits,
            symmetric=self.symmetric,
            per_channel=True,
            use_ste=self.training,
        )
        w_restored = w_q / s

        b_q = None
        if self.bias is not None:
            b_q = quantize_dequantize_tensor(
                self.bias,
                bits=self.bits,
                symmetric=self.symmetric,
                per_channel=False,
                use_ste=self.training,
            )
        return nn.functional.linear(x, w_restored, b_q)


class FAQQuantizer(BaseCompressor):
    """
    FAQQuantizer: Feature-Activation-Aware Quantization.
    Collects activation statistics to scale channel-wise quantization step sizes,
    minimizing activation-weighted reconstruction distortion.

    Args:
        model: PyTorch model.
        tokenizer: Optional tokenizer for text-based calibration.
        bits: Quantization bit-width (4 or 8). Default 8.
        calibration_texts: List of text strings for calibration.
        calibration_data: Optional Tensor or DataLoader for calibration.
        alpha: Activation scaling exponent. Default 0.5.
        symmetric: Whether to use symmetric quantization. Default True.
        use_ste: Whether to substitute nn.Linear with FAQLinear. Default False.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        bits: int = 8,
        calibration_texts: Optional[List[str]] = None,
        calibration_data: Optional[Any] = None,
        alpha: float = 0.5,
        symmetric: bool = True,
        use_ste: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="faq", **kwargs)
        self.tokenizer = tokenizer
        self.bits = int(bits)
        self.calibration_texts = calibration_texts
        self.calibration_data = calibration_data
        self.alpha = float(alpha)
        self.symmetric = bool(symmetric)
        self.use_ste = bool(use_ste)

    def _collect_activation_scales(self) -> Dict[str, torch.Tensor]:
        """Collect channel-wise input activation scales per Linear layer via forward hooks."""
        act_scales: Dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(layer_name: str) -> Any:
            def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], outp: Any) -> None:
                if len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                    x = inp[0].detach().float()
                    # Compute mean/max magnitude per input channel (last dimension)
                    if x.ndim >= 2:
                        flat_x = x.reshape(-1, x.shape[-1])
                        channel_scale = torch.mean(torch.abs(flat_x), dim=0)
                    else:
                        channel_scale = torch.abs(x)
                    if layer_name in act_scales:
                        act_scales[layer_name] = 0.5 * (act_scales[layer_name] + channel_scale.cpu())
                    else:
                        act_scales[layer_name] = channel_scale.cpu()

            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(name)))

        self.model.eval()
        with torch.no_grad():
            try:
                if self.tokenizer is not None and (self.calibration_texts or hasattr(self.model, "tokenizer")):
                    texts = self.calibration_texts or ["The quick brown fox jumps over the lazy dog"]
                    inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
                    first_p = next(self.model.parameters(), None)
                    if first_p is not None and first_p.is_cuda:
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    self.model(**inputs)
                elif self.calibration_data is not None:
                    if isinstance(self.calibration_data, torch.Tensor):
                        first_p = next(self.model.parameters(), None)
                        dev = first_p.device if first_p is not None else torch.device("cpu")
                        self.model(self.calibration_data.to(dev))
                    elif hasattr(self.calibration_data, "__iter__"):
                        for batch in self.calibration_data:
                            if isinstance(batch, torch.Tensor):
                                self.model(batch)
                            elif isinstance(batch, dict):
                                self.model(**batch)
                            break
                else:
                    # Synthetic calibration forward pass
                    first_linear = next((m for m in self.model.modules() if isinstance(m, nn.Linear)), None)
                    if first_linear is not None:
                        first_p = next(self.model.parameters(), None)
                        dev = first_p.device if first_p is not None else torch.device("cpu")
                        dummy_x = torch.randn(2, 16, first_linear.in_features, device=dev)
                        try:
                            self.model(dummy_x)
                        except Exception:
                            # If model doesn't accept tensor directly, run per-layer synthetic
                            for name, module in self.model.named_modules():
                                if isinstance(module, nn.Linear):
                                    act_scales[name] = torch.ones(module.in_features)
            except Exception:
                # Fallback to uniform activation scales
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Linear):
                        act_scales[name] = torch.ones(module.in_features)

        for h in hooks:
            h.remove()

        # Ensure all linear layers have an entry
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name not in act_scales:
                act_scales[name] = torch.ones(module.in_features)

        return act_scales

    def _replace_module(self, full_name: str, new_module: nn.Module) -> None:
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Quantize model weights using FAQ activation statistics."""
        act_scales = self._collect_activation_scales()

        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear):
                s_act = act_scales.get(name, torch.ones(module.in_features)).to(module.weight.device)
                s_pow = torch.pow(torch.clamp(s_act, min=1e-6), self.alpha).unsqueeze(0)

                if self.use_ste:
                    faq_layer = FAQLinear.from_linear(
                        module,
                        act_scale=s_act,
                        bits=self.bits,
                        symmetric=self.symmetric,
                    )
                    self._replace_module(name, faq_layer)
                else:
                    # Scale weight by activation scale, quantize, and scale back
                    W = module.weight.data.float()
                    W_scaled = W * s_pow
                    W_q = quantize_dequantize_tensor(
                        W_scaled,
                        bits=self.bits,
                        symmetric=self.symmetric,
                        per_channel=True,
                        use_ste=False,
                    )
                    module.weight.data = (W_q / s_pow).to(dtype=module.weight.dtype)

                    if module.bias is not None:
                        module.bias.data = quantize_dequantize_tensor(
                            module.bias.data,
                            bits=self.bits,
                            symmetric=self.symmetric,
                            per_channel=False,
                            use_ste=False,
                        )

        stats = self.get_compression_stats()
        stats["quantization_bits"] = self.bits
        stats["alpha"] = self.alpha
        stats["symmetric"] = self.symmetric
        stats["theoretical_vram_saving"] = f"{(1.0 - self.bits / 16.0) * 100:.1f}% vs FP16"
        return self.model, stats


@COMPRESSOR_REGISTRY.register("faq")
def faq_factory(model: nn.Module, comp_config: Any = None) -> nn.Module:
    """FAQ compressor factory function."""
    bits = getattr(comp_config, "faq_bits", 8) if comp_config else 8
    alpha = getattr(comp_config, "faq_alpha", 0.5) if comp_config else 0.5
    tokenizer = getattr(model, "tokenizer", None)
    compressor = FAQQuantizer(model, tokenizer=tokenizer, bits=bits, alpha=alpha)
    compressed_model, _ = compressor.compress()
    return compressed_model
