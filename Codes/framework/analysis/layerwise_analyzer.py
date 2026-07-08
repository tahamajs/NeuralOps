"""Layer‑wise analysis: cosine similarity, effective rank, intrinsic dimension."""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy.spatial.distance import cosine


class LayerwiseAnalyzer:
    def __init__(self, model, tokenizer, device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.hidden_states = []
        self.hooks = []

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def register_hooks(self):
        self.hidden_states = []

        def hook_fn(module, input, output):
            hs = (
                output[0].detach().cpu()
                if isinstance(output, tuple)
                else output.detach().cpu()
            )
            self.hidden_states.append(hs)

        for name, module in self.model.named_modules():
            if "encoder.layer" in name or "transformer.layer" in name:
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def analyze(self, texts: List[str], batch_size: int = 8) -> Dict[str, Any]:
        self.register_hooks()
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                self.model(**inputs)
            # all_hidden.append(self.hidden_states[-1])  # last layer only? We'll collect all
        # Stack hidden states per layer
        num_layers = len(self.hidden_states)
        avg_hidden_per_layer = []
        for layer_idx in range(num_layers):
            stacked = self.hidden_states[layer_idx].float().mean(dim=(0, 1)).numpy()
            avg_hidden_per_layer.append(stacked)
        self.remove_hooks()
        # Cosine similarities between consecutive layers
        similarities = []
        for i in range(len(avg_hidden_per_layer) - 1):
            sim = 1 - cosine(avg_hidden_per_layer[i], avg_hidden_per_layer[i + 1])
            similarities.append(max(0, min(1, sim)))
        return {
            "num_layers": num_layers,
            "cosine_similarities": similarities,
            "effective_ranks": [],  # placeholder
        }


def estimate_intrinsic_dimension(hidden_states: np.ndarray, k: int = 5) -> float:
    N, D = hidden_states.shape
    if N < k + 1:
        return float(D)
    from scipy.spatial.distance import pdist, squareform

    dists = squareform(pdist(hidden_states))
    np.fill_diagonal(dists, np.inf)
    kth_dists = np.sort(dists, axis=1)[:, k - 1]
    valid = kth_dists > 0
    if not valid.any():
        return float(D)
    log_ratios = np.log(kth_dists[valid] / kth_dists[valid][:, None])
    id_estimate = -1.0 / np.mean(log_ratios)
    return max(1.0, min(float(D), id_estimate))
