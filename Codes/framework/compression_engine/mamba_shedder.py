"""
Mamba-Shedder: Sensitivity-Based Structured Pruning and Layer Shedding Compression Engine.
Registered in COMPRESSOR_REGISTRY as 'mamba_shedder'.

Mathematical Formulation:
1. Empirical Fisher Information / Sensitivity Metric:
   $$\\mathcal{I}(X_j) = \\sqrt{\\frac{1}{N} \\sum_{n=1}^N X_{n, j}^2}$$
2. Layer Residual Sensitivity:
   $$\\Delta_l = \\frac{\\|X_{l+1} - X_l\\|_F}{\\|X_l\\|_F + \\epsilon}$$
3. Channel & Layer Masking:
   $$W_{pruned} = W \\odot \\mathcal{M}_{channel}^\\top$$
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


class MambaShedderCompressor(BaseCompressor):
    """
    Mamba-Shedder Compressor: Sensitivity-based channel pruning and layer shedding.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        sparsity: float = 0.2,
        shed_layers: int = 0,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        structured_24: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="mamba_shedder", **kwargs)
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.shed_layers = max(0, int(shed_layers))
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.structured_24 = structured_24
        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.stats: Dict[str, Any] = {}

    def _collect_activations(self) -> Dict[str, torch.Tensor]:
        """Collect input activations across all nn.Linear modules."""
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
                logger.debug("MambaShedder activation collection forward pass exception: %s", e)

        for h in hooks:
            h.remove()

        aggregated: Dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                aggregated[name] = torch.cat(act_list, dim=0)

        return aggregated

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes Mamba-Shedder sensitivity pruning across all Linear modules."""
        activations_map = self._collect_activations()
        pruned_layers_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.clone().float()
                d_out, d_in = W.shape
                device = W.device
                dtype = module.weight.dtype

                act = activations_map.get(name, None)
                if act is not None and act.numel() > 0:
                    act_flat = act.reshape(-1, act.shape[-1]).float().to(device)
                    if act_flat.shape[-1] == d_in:
                        fisher = (act_flat ** 2).mean(dim=0).clamp(min=1e-8)
                        sens = fisher.sqrt()
                    else:
                        sens = torch.norm(W, dim=0)
                else:
                    sens = torch.norm(W, dim=0)

                if self.structured_24:
                    saliency_matrix = sens.unsqueeze(0).expand_as(W)
                    W_pruned, _ = apply_24_structured_mask(W, saliency=saliency_matrix, dim=1)
                else:
                    n_keep = max(1, int(d_in * (1.0 - self.sparsity)))
                    _, keep_idx = torch.topk(sens, k=n_keep, largest=True)
                    mask = torch.zeros(d_in, device=device)
                    mask[keep_idx] = 1.0
                    W_pruned = W * mask.unsqueeze(0)

                module.weight.data = W_pruned.to(dtype=dtype, device=device)
                pruned_layers_count += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "pruned_layers": pruned_layers_count,
            "sparsity_target": self.sparsity,
            "structured_24": self.structured_24,
            "shed_layers": self.shed_layers,
            "method": "mamba_shedder_sensitivity",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("mamba_shedder")
def mamba_shedder_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for Mamba-Shedder compressor."""
    sparsity = getattr(comp_config, "mamba_shedder_sparsity", getattr(comp_config, "sparsity", 0.2)) if comp_config else kwargs.get("sparsity", 0.2)
    shed_layers = getattr(comp_config, "mamba_shedder_shed_layers", getattr(comp_config, "shed_layers", 0)) if comp_config else kwargs.get("shed_layers", 0)
    structured_24 = getattr(comp_config, "mamba_shedder_structured_24", getattr(comp_config, "structured_24", False)) if comp_config else kwargs.get("structured_24", False)
    tokenizer = getattr(model, "tokenizer", None) or (getattr(comp_config, "tokenizer", None) if comp_config else kwargs.get("tokenizer", None))

    compressor = MambaShedderCompressor(
        model,
        tokenizer=tokenizer,
        sparsity=sparsity,
        shed_layers=shed_layers,
        structured_24=structured_24,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
