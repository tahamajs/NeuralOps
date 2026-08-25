# SwiftSVDActivationCompressor: Activation-Aware SVD via Cross-Covariance Projection

## 1. Executive Abstract & Theoretical Motivation
Pure weight-based SVD compresses $W$ without accounting for the distribution of activations that pass through the layer during inference. Activation-aware compression seeks low-rank factors that minimize output distortion:

$$\min_{A, B} \mathbb{E}_{x \sim \mathcal{D}} \|W x - A B x\|_2^2$$

`SwiftSVDActivationCompressor` (registered as `swift_svd_act`) solves this efficiently by capturing input activations $X \in \mathbb{R}^{N \times d_{\text{in}}}$ and output activations $Y = X W^T \in \mathbb{R}^{N \times d_{\text{out}}}$ over a small calibration dataset. It forms the empirical cross-covariance matrix $C_{XY} = \frac{1}{N} Y^T X$, decomposes $C_{XY}$ via SVD, distributes singular values symmetrically ($U \sqrt{\Sigma}, \sqrt{\Sigma} V^T$), and factorizes dense `nn.Linear` layers into structural `SVDLinear` modules.

---

## 2. Mathematical Formulations

### 2.1 Empirical Cross-Covariance Estimation
Given calibration inputs streamed through the network, forward hooks capture:
- Input activations: $X \in \mathbb{R}^{N \times d_{\text{in}}}$ (flattened over batch and sequence tokens).
- Output activations: $Y \in \mathbb{R}^{N \times d_{\text{out}}}$.

The sample input-output cross-covariance matrix $C_{XY} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is accumulated as:

$$C_{XY} = \frac{1}{N} Y^T X$$

Because $Y \approx X W^T$, $C_{XY} = \frac{1}{N} W X^T X = W C_{XX}$, representing the weight matrix pre-conditioned by the input activation covariance $C_{XX}$.

### 2.2 Cross-Covariance SVD
Applying SVD to the cross-covariance matrix:

$$C_{XY} = U \Sigma V^T$$

where:
- Left singular vectors $U \in \mathbb{R}^{d_{\text{out}} \times d_{\text{out}}}$ span the principal output activation directions.
- Right singular vectors $V^T \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ span the principal input activation directions.
- $\Sigma = \operatorname{diag}(\sigma_1, \dots, \sigma_{\min(d_{\text{in}}, d_{\text{out}})})$ contains cross-spectral energies.

### 2.3 Symmetrical Square-Root Factorization
The top-$r$ components are retained with balanced singular value scaling:

$$U_r = U[:, :r] \operatorname{diag}(\sqrt{\Sigma_{:r}}) \in \mathbb{R}^{d_{\text{out}} \times r}$$

$$V_r = \operatorname{diag}(\sqrt{\Sigma_{:r}}) V^T[:r, :] \in \mathbb{R}^{r \times d_{\text{in}}}$$

The layer is replaced with `SVDLinear`, executing forward propagation as:

$$y = (x V_r^T) U_r^T + b$$

---

## 3. Architecture & Calibration Dataflow

```
                Calibration Input Batches
                            │
                            ▼
              ┌───────────────────────────┐
              │ Forward Pass with Hooks   │
              │ Capture (x_in, y_out)     │
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │ Cross-Covariance Matrix   │
              │ C_XY = (Y^T @ X) / N      │
              │ Shape: [d_out, d_in]      │
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │ Truncated SVD on C_XY     │
              │ U_r sqrt(S), sqrt(S) V_r^T│
              └─────────────┬─────────────┘
                            │  [Low-rank tensors]
                            ▼
              ┌───────────────────────────┐
              │ SVDLinear Layer Injection │
              │ y = (x @ V_r.T) @ U_r.T   │
              └───────────────────────────┘
```

---

## 4. API Reference & Factory Usage

### 4.1 Class Reference
```python
from framework.compression_engine.swift_svd_act import SwiftSVDActivationCompressor

compressor = SwiftSVDActivationCompressor(
    model=model,
    calibration_loader=val_loader,   # Calibration DataLoader
    rank=32,                         # Target bottleneck rank r
    target_modules=None,             # Optional layer filters
)

compressed_model, stats = compressor.compress()
```

### 4.2 Factory Functions
```python
from framework.compression_engine.swift_svd_act import (
    swift_svd_act_factory,
    swift_svd_act_compressor_factory,
)

# Universal factory signature
compressed_model = swift_svd_act_factory(
    model=model,
    comp_config=None,
    rank=32,
    calibration_loader=val_loader,
)
```

### 4.3 Parameters
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `nn.Module` | Required | PyTorch neural network to compress. |
| `rank` | `int` | `64` | Target bottleneck rank $r$. |
| `calibration_loader` | `DataLoader` | `None` | Optional DataLoader providing calibration batches. |
| `calibration_texts` | `List[str]` | `None` | Optional text strings for NLP calibration. |
| `tokenizer` | `Any` | `None` | Optional tokenizer for tokenizing `calibration_texts`. |
| `target_modules` | `List[str]` | `None` | Layer name substring filter. |

---

## Complete Source Code Implementation (`swift_svd_act.py`)

```python
"""
Swift-SVD Activation-Aware Compressor via cross-covariance / activation projection.
Registered as 'swift_svd_act'.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("swift_svd_act")
class SwiftSVDActivationCompressor(BaseCompressor):
    """
    Swift-SVD Activation-Aware Compressor.

    Captures empirical input-output cross-covariances (C_XY = Y^T X / N) across calibration data,
    computes activation-aligned singular subspaces, performs symmetric square-root singular
    value splitting (U_r * sqrt(Sigma_r), sqrt(Sigma_r) * V_r^T), and replaces dense nn.Linear
    layers with factorized SVDLinear modules.

    Args:
        model: Model to compress.
        tokenizer: Optional tokenizer for text calibration.
        calibration_loader: Optional DataLoader for calibration.
        calibration_texts: Optional list of calibration text strings.
        rank: Target low-rank bottleneck dimension r.
        target_modules: Optional list of module name filters.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        rank: int = 64,
        target_modules: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="swift_svd_act", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = getattr(cfg, "swift_svd_act_rank", getattr(cfg, "rank", rank))
            tokenizer = getattr(cfg, "tokenizer", tokenizer)
            calibration_loader = getattr(cfg, "calibration_loader", calibration_loader)
            calibration_texts = getattr(cfg, "calibration_texts", calibration_texts)

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]

        self.tokenizer = tokenizer
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.rank = int(rank)
        self.target_modules = target_modules

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if isinstance(module, SVDLinear):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def _collect_activations(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Collect input and output activation tensors from calibration passes."""
        activations: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inp: Tuple[Any, ...], outp: Any) -> None:
                if not inp or inp[0] is None:
                    return
                x_in = inp[0].detach()
                x_out = outp.detach() if isinstance(outp, torch.Tensor) else outp[0].detach()
                activations[name] = (x_in, x_out)

            return hook

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            return {}

        for name, mod in target_layers:
            hooks.append(mod.register_forward_hook(make_hook(name)))

        device = next(self.model.parameters()).device
        self.model.eval()

        try:
            with torch.no_grad():
                if self.calibration_loader is not None:
                    for batch in self.calibration_loader:
                        if isinstance(batch, dict):
                            batch = {
                                k: v.to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in batch.items()
                            }
                            self.model(**batch)
                        elif isinstance(batch, (list, tuple)):
                            batch = [
                                v.to(device) if isinstance(v, torch.Tensor) else v
                                for v in batch
                            ]
                            self.model(*batch)
                        elif isinstance(batch, torch.Tensor):
                            self.model(batch.to(device))
                        break
                elif self.tokenizer is not None and self.calibration_texts is not None:
                    inputs = self.tokenizer(
                        self.calibration_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    self.model(**inputs)
        except Exception as e:
            logger.warning(
                "SwiftSVDActivation: Calibration forward failed (%s). Falling back to standard SVD.",
                e,
            )
        finally:
            for h in hooks:
                h.remove()

        return activations

    def compress_linear(
        self,
        layer: nn.Linear,
        act_pair: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> SVDLinear:
        """Compress a single linear layer using activation cross-covariance or weight SVD."""
        out_f = layer.out_features
        in_f = layer.in_features
        r = max(1, min(self.rank, out_f, in_f))
        device = layer.weight.device
        dtype = layer.weight.dtype

        if act_pair is not None:
            inp, out = act_pair
            inp_flat = inp.reshape(-1, inp.size(-1)).float().to(device)
            out_flat = out.reshape(-1, out.size(-1)).float().to(device)

            # Cross-covariance matrix C_XY: [out_features, in_features]
            n_tokens = max(1, inp_flat.size(0))
            C = (out_flat.t() @ inp_flat) / n_tokens

            try:
                U, S, Vt = torch.linalg.svd(C, full_matrices=False)
            except Exception:
                W = layer.weight.data.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        else:
            W = layer.weight.data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)

        # Symmetric square-root singular value distribution
        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
        V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(dtype=dtype, device=device)

        svd_layer = SVDLinear(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=layer.bias is not None,
            dtype=dtype,
            device=device,
        )
        svd_layer.set_low_rank_factors(U_r, V_r)
        if layer.bias is not None:
            svd_layer.bias.data.copy_(layer.bias.data)

        return svd_layer

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress all linear layers in the model."""
        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("SwiftSVDActivation: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        activations = self._collect_activations()

        for name, module in target_layers:
            act_pair = activations.get(name, None)
            svd_layer = self.compress_linear(module, act_pair=act_pair)
            replace_module(self.model, name, svd_layer)

        stats = self.get_compression_stats()
        return self.model, stats


def swift_svd_act_factory(
    model: nn.Module, comp_config: Any = None, **kwargs: Any
) -> nn.Module:
    """Swift-SVD activation-aware factory function."""
    compressor = SwiftSVDActivationCompressor(model, comp_config=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
```
