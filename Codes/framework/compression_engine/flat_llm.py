"""
FLAT-LLM: Activation-Aware Structured Dimension & Neuron Pruning via Eigen-Decomposition.
Registered in COMPRESSOR_REGISTRY as 'flat_llm'.

Mathematical Formulation:
1. Output Activation Covariance:
   $$\\Sigma_Y = \\frac{1}{N} Y^\\top Y \\in \\mathbb{R}^{d_{out} \\times d_{out}}$$
2. Eigendecomposition:
   $$\\Sigma_Y = V \\Lambda V^\\top, \\quad \\Lambda = \\operatorname{diag}(\\lambda_1, \\dots, \\lambda_{d_{out}})$$
3. Retain top $k = \\lfloor d_{out} \\cdot \\text{keep\\_ratio} \\rfloor$ most informative dimensions:
   $$\\mathcal{I}_{kept} = \\operatorname{Top-}k(\\Lambda)$$
4. Output Neuron Mask:
   $$\\mathcal{M}_i = \\begin{cases} 1 & \\text{if } i \\in \\mathcal{I}_{kept} \\\\ 0 & \\text{otherwise} \\end{cases}$$
5. Pruned Weights & Biases:
   $$W_{flat} = \\mathcal{M} \\odot W, \\quad b_{flat} = \\mathcal{M} \\odot b$$
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


class FLATLLMCompressor(BaseCompressor):
    """
    FLAT-LLM Compressor: Activation-aware neuron and output channel pruning.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        keep_ratio: float = 0.7,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        structured_24: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="flat_llm", **kwargs)
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.keep_ratio = max(0.05, min(float(keep_ratio), 1.0))
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.structured_24 = structured_24
        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.stats: Dict[str, Any] = {}

    def _collect_layer_outputs(self) -> Dict[str, torch.Tensor]:
        """Capture output activations across all nn.Linear modules."""
        self.model.eval()
        activations: Dict[str, List[torch.Tensor]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any):
                if isinstance(output, torch.Tensor):
                    out = output.detach()
                    if name not in activations:
                        activations[name] = []
                    activations[name].append(out.cpu())
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
                logger.debug("FLAT-LLM output collection forward pass exception: %s", e)

        for h in hooks:
            h.remove()

        aggregated: Dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                aggregated[name] = torch.cat(act_list, dim=0)

        return aggregated

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes FLAT-LLM dimension pruning across all Linear modules."""
        outputs_map = self._collect_layer_outputs()
        pruned_layers_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.clone().float()
                d_out, d_in = W.shape
                device = W.device
                dtype = module.weight.dtype

                act = outputs_map.get(name, None)
                if act is not None and act.numel() > 0:
                    act_flat = act.reshape(-1, act.shape[-1]).float().to(device)
                    if act_flat.shape[-1] == d_out:
                        cov = torch.matmul(act_flat.T, act_flat) / max(act_flat.shape[0], 1)
                        cov = cov + 1e-6 * torch.eye(d_out, device=device)
                        eigvals, _ = torch.linalg.eigh(cov)
                    else:
                        eigvals = torch.sum(W.pow(2), dim=1)
                else:
                    eigvals = torch.sum(W.pow(2), dim=1)

                if self.structured_24:
                    saliency_matrix = eigvals.unsqueeze(1).expand_as(W)
                    W_pruned, _ = apply_24_structured_mask(W, saliency=saliency_matrix, dim=0)
                    module.weight.data = W_pruned.to(dtype=dtype, device=device)
                else:
                    keep_dims = max(1, int(d_out * self.keep_ratio))
                    _, keep_idx = torch.topk(eigvals, k=keep_dims, largest=True)
                    mask = torch.zeros(d_out, device=device)
                    mask[keep_idx] = 1.0

                    W_pruned = W * mask.unsqueeze(1)
                    module.weight.data = W_pruned.to(dtype=dtype, device=device)
                    if module.bias is not None:
                        module.bias.data = (module.bias.data * mask).to(dtype=dtype, device=device)

                pruned_layers_count += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "pruned_layers": pruned_layers_count,
            "keep_ratio": self.keep_ratio,
            "structured_24": self.structured_24,
            "method": "flat_llm_eigen_pruning",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("flat_llm")
@COMPRESSOR_REGISTRY.register("flat_llm_compressor")
def flat_llm_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for FLAT-LLM compressor."""
    keep_ratio = getattr(comp_config, "flat_llm_keep_ratio", getattr(comp_config, "keep_ratio", 0.7)) if comp_config else kwargs.get("keep_ratio", 0.7)
    structured_24 = getattr(comp_config, "flat_llm_structured_24", getattr(comp_config, "structured_24", False)) if comp_config else kwargs.get("structured_24", False)
    tokenizer = getattr(model, "tokenizer", None) or (getattr(comp_config, "tokenizer", None) if comp_config else kwargs.get("tokenizer", None))

    compressor = FLATLLMCompressor(
        model,
        tokenizer=tokenizer,
        keep_ratio=keep_ratio,
        structured_24=structured_24,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
