"""FLAP: Fluctuation-based Adaptive Structured Pruning. Registered as 'flap'."""

import numpy as np
import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class FLAPCompressor(BaseCompressor):
    def __init__(
        self, model, tokenizer, sparsity=0.2, device="cuda", nsamples=1024, **kwargs
    ):
        super().__init__(model, name="flap", **kwargs)
        self.tokenizer = tokenizer
        self.sparsity = sparsity
        self.device = device
        self.nsamples = nsamples

    def _compute_fluctuation_metric(self, weight, inputs, metric_type="WIFV"):
        inp = inputs.detach().float()
        inp_var = torch.var(inp, dim=0)
        if metric_type == "IFV":
            scores = inp_var
        elif metric_type == "WIFV":
            scores = inp_var * torch.sum(weight.pow(2), dim=0)
        elif metric_type == "WIFN":
            scores = (torch.abs(weight) * torch.sqrt(inp_var.reshape(1, -1))).mean(
                dim=0
            )
        else:
            raise ValueError(f"Unknown metric: {metric_type}")
        return scores.cpu().numpy()

    def compress(self):
        print(f"✂️ Applying FLAP (sparsity={self.sparsity})...")
        self.model.eval()
        # Simple calibration set
        calib_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subfield of artificial intelligence.",
            "Transformers have revolutionized natural language processing.",
        ] * (self.nsamples // 3)
        inputs = self.tokenizer(
            calib_texts, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs.input_ids.to(self.device)
        with torch.no_grad():
            embeds = self.model.get_input_embeddings()(input_ids)
            flat_inputs = embeds.view(-1, embeds.shape[-1])

        for layer_idx, layer in enumerate(self.model.encoder.layer):
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    weight = module.weight.data
                    scores = self._compute_fluctuation_metric(weight, flat_inputs)
                    n_keep = int(weight.shape[1] * (1 - self.sparsity))
                    keep_idx = np.argsort(scores)[-n_keep:]
                    # Soft pruning: zero out unimportant columns
                    mask = torch.zeros(weight.shape[1], device=weight.device)
                    mask[keep_idx] = 1.0
                    module.weight.data = weight * mask.unsqueeze(
                        0
                    )  # broadcast over rows
                    # Bias remains untouched (no pruning)
                    # Update flat_inputs to reflect the pruned feature distribution (optional)
                    flat_inputs = flat_inputs[:, keep_idx]
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("flap")
def flap_compressor_factory(model, comp_config):
    sparsity = getattr(comp_config, "flap_sparsity", 0.2)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("flap requires tokenizer attached as model.tokenizer")
    compressor = FLAPCompressor(model, tokenizer, sparsity=sparsity)
    compressed_model, _ = compressor.compress()
    return compressed_model
