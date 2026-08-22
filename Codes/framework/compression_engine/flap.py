"""
FLAP: Fluctuation-Based Adaptive Structured Pruning.
Registered in COMPRESSOR_REGISTRY as 'flap'.

Mathematical Formulation:
1. Input Feature Activation Variance:
   $$\\text{Var}(X_j) = \\frac{1}{N} \\sum_{n=1}^N (X_{n, j} - \\bar{X}_j)^2$$
2. Saliency Metrics:
   - **IFV** (Input Feature Variance): $S_j = \\text{Var}(X_j)$
   - **WIFV** (Weight-Integrated Fluctuation Variance): $S_j = \\text{Var}(X_j) \\sum_{i=1}^{d_{out}} W_{ij}^2$
   - **WIFN** (Weight-Integrated Fluctuation Norm): $S_j = \\|W_{:, j}\\|_1 \\sqrt{\\text{Var}(X_j)}$
3. Structured Column Mask:
   $$\\mathcal{M}_j = \\begin{cases} 1 & \\text{if } S_j \\in \\operatorname{Top-}(1 - s)(S) \\\\ 0 & \\text{otherwise} \\end{cases}$$
4. Pruned Weights:
   $$W_{flap} = W \\odot \\mathcal{M}^\\top$$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import (
    apply_24_structured_mask,
    get_calibration_tensors,
)

logger = logging.getLogger(__name__)


class FLAPCompressor(BaseCompressor):
    """
    FLAP Compressor: Fluctuation-based Adaptive Structured Pruning.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        sparsity: float = 0.2,
        metric_type: str = "WIFV",
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        structured_24: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="flap", **kwargs)
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.metric_type = metric_type.upper()
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.structured_24 = structured_24
        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.stats: Dict[str, Any] = {}

    def _collect_layer_activations(self) -> Dict[str, torch.Tensor]:
        """Capture layer input activations across all nn.Linear modules."""
        self.model.eval()
        activations: Dict[str, List[torch.Tensor]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    inp = inputs[0].detach()
                    if name not in activations:
                        activations[name] = []
                    activations[name].append(inp.cpu())
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(name)))

        calib_inputs = get_calibration_tensors(
            self.model,
            tokenizer=self.tokenizer,
            calibration_loader=self.calibration_loader,
            calibration_texts=self.calibration_texts,
            device=self.target_device,
        )

        with torch.no_grad():
            try:
                if isinstance(calib_inputs, dict):
                    self.model(**calib_inputs)
                elif isinstance(calib_inputs, torch.Tensor):
                    self.model(calib_inputs)
                elif isinstance(calib_inputs, (list, tuple)):
                    self.model(*calib_inputs)
            except Exception as e:
                logger.debug("FLAP activation collection forward pass exception: %s", e)

        for h in hooks:
            h.remove()

        aggregated: Dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                aggregated[name] = torch.cat(act_list, dim=0)

        return aggregated

    def _compute_fluctuation_metric(
        self,
        weight: torch.Tensor,
        activations: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Computes per-column fluctuation saliency score."""
        d_out, d_in = weight.shape
        device = weight.device

        if activations is not None and activations.numel() > 0:
            X = activations.reshape(-1, activations.shape[-1]).float().to(device)
            if X.shape[-1] == d_in:
                inp_var = torch.var(X, dim=0).clamp(min=1e-8)  # [d_in]
            else:
                inp_var = torch.ones(d_in, device=device)
        else:
            inp_var = torch.ones(d_in, device=device)

        W = weight.float()
        if self.metric_type == "IFV":
            scores = inp_var
        elif self.metric_type == "WIFV":
            scores = inp_var * torch.sum(W.pow(2), dim=0)
        elif self.metric_type == "WIFN":
            scores = torch.sum(torch.abs(W), dim=0) * torch.sqrt(inp_var)
        else:
            scores = inp_var * torch.sum(W.pow(2), dim=0)

        return scores

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes FLAP structured column pruning across all Linear modules."""
        activations_map = self._collect_layer_activations()
        pruned_layers_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.clone().float()
                device = W.device
                dtype = module.weight.dtype

                act = activations_map.get(name, None)
                scores = self._compute_fluctuation_metric(W, act)

                if self.structured_24:
                    saliency_matrix = scores.unsqueeze(0).expand_as(W)
                    W_pruned, _ = apply_24_structured_mask(W, saliency=saliency_matrix, dim=1)
                else:
                    n_keep = max(1, int(W.shape[1] * (1.0 - self.sparsity)))
                    _, keep_idx = torch.topk(scores, k=n_keep, largest=True)
                    mask = torch.zeros(W.shape[1], device=device)
                    mask[keep_idx] = 1.0
                    W_pruned = W * mask.unsqueeze(0)

                module.weight.data = W_pruned.to(dtype=dtype, device=device)
                pruned_layers_count += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "pruned_layers": pruned_layers_count,
            "metric_type": self.metric_type,
            "sparsity_target": self.sparsity,
            "structured_24": self.structured_24,
            "method": "flap_fluctuation_pruning",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("flap")
@COMPRESSOR_REGISTRY.register("flap_compressor")
def flap_compressor_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for FLAP compressor."""
    sparsity = getattr(comp_config, "flap_sparsity", getattr(comp_config, "sparsity", 0.2)) if comp_config else kwargs.get("sparsity", 0.2)
    metric_type = getattr(comp_config, "flap_metric_type", getattr(comp_config, "metric_type", "WIFV")) if comp_config else kwargs.get("metric_type", "WIFV")
    structured_24 = getattr(comp_config, "flap_structured_24", getattr(comp_config, "structured_24", False)) if comp_config else kwargs.get("structured_24", False)
    tokenizer = getattr(model, "tokenizer", None) or (getattr(comp_config, "tokenizer", None) if comp_config else kwargs.get("tokenizer", None))

    compressor = FLAPCompressor(
        model,
        tokenizer=tokenizer,
        sparsity=sparsity,
        metric_type=metric_type,
        structured_24=structured_24,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
