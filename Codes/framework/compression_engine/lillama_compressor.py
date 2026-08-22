"""
LiLLaMA: Local Distillation via Activation Cross-Covariance Singular Value Decomposition.
Registered in COMPRESSOR_REGISTRY as 'lillama'.

Mathematical Formulation:
1. Input and Output Activations:
   $$X \\in \\mathbb{R}^{N \\times d_{in}}, \\quad Y \\in \\mathbb{R}^{N \\times d_{out}}$$
2. Empirical Cross-Covariance Matrix:
   $$C_{YX} = \\frac{1}{N} Y^\\top X \\in \\mathbb{R}^{d_{out} \\times d_{in}}$$
3. Truncated Singular Value Decomposition:
   $$C_{YX} = U \\Sigma V^\\top \\approx U_r \\Sigma_r V_r^\\top$$
4. Optimal Distilled Weight Reconstruction:
   $$W_{distill} = U_r \\Sigma_r V_r^\\top$$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import get_calibration_tensors

logger = logging.getLogger(__name__)


class LiLLaMACompressor(BaseCompressor):
    """
    LiLLaMA Compressor: Local activation cross-covariance SVD distillation.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        rank: int = 32,
        calibration_texts: Optional[List[str]] = None,
        calibration_loader: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="lillama", **kwargs)
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.rank = max(1, int(rank))
        self.calibration_texts = calibration_texts
        self.calibration_loader = calibration_loader
        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.stats: Dict[str, Any] = {}

    def _collect_io_activations(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Collect input and output activations across all Linear layers."""
        self.model.eval()
        activations: Dict[str, Dict[str, List[torch.Tensor]]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any):
                inp = inputs[0].detach() if inputs and isinstance(inputs[0], torch.Tensor) else None
                out = output.detach() if isinstance(output, torch.Tensor) else (output[0].detach() if isinstance(output, (tuple, list)) else None)
                if inp is not None and out is not None:
                    if name not in activations:
                        activations[name] = {"in": [], "out": []}
                    activations[name]["in"].append(inp.cpu())
                    activations[name]["out"].append(out.cpu())
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
                logger.debug("LiLLaMA IO activation collection forward pass exception: %s", e)

        for h in hooks:
            h.remove()

        aggregated: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name, data in activations.items():
            if data["in"] and data["out"]:
                cat_in = torch.cat(data["in"], dim=0)
                cat_out = torch.cat(data["out"], dim=0)
                aggregated[name] = (cat_in, cat_out)

        return aggregated

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes LiLLaMA cross-covariance SVD distillation."""
        activations_map = self._collect_io_activations()
        distilled_layers_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.clone().float()
                d_out, d_in = W.shape
                device = W.device
                dtype = module.weight.dtype

                if name in activations_map:
                    inp, out = activations_map[name]
                    inp_flat = inp.reshape(-1, inp.shape[-1]).float()
                    out_flat = out.reshape(-1, out.shape[-1]).float()

                    if inp_flat.shape[-1] == d_in and out_flat.shape[-1] == d_out:
                        # Cross-covariance
                        C = torch.matmul(out_flat.T, inp_flat) / max(len(inp_flat), 1)
                        try:
                            U, S, Vh = torch.linalg.svd(C, full_matrices=False)
                            r = min(self.rank, min(d_out, d_in), len(S))
                            W_new = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vh[:r, :])
                        except Exception:
                            W_new = W
                    else:
                        W_new = W
                else:
                    # SVD on weight itself as fallback
                    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                    r = min(self.rank, len(S))
                    W_new = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vh[:r, :])

                module.weight.data = W_new.to(dtype=dtype, device=device)
                distilled_layers_count += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "distilled_layers": distilled_layers_count,
            "rank": self.rank,
            "method": "lillama_cross_covariance_svd",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("lillama")
@COMPRESSOR_REGISTRY.register("lillama_compressor")
def lillama_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for LiLLaMA compressor."""
    rank = getattr(comp_config, "lillama_rank", getattr(comp_config, "rank", 32)) if comp_config else kwargs.get("rank", 32)
    tokenizer = getattr(model, "tokenizer", None) or (getattr(comp_config, "tokenizer", None) if comp_config else kwargs.get("tokenizer", None))

    compressor = LiLLaMACompressor(model, tokenizer=tokenizer, rank=rank, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
