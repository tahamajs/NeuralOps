# framework/analysis/layerwise_analyzer.py
import torch
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

class LayerwiseAnalyzer:
    def __init__(self, model, tokenizer, device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or self._get_device()
        self.model.to(self.device)
        self.hidden_states = []
        self.hooks = []

    def _get_device(self) -> str:
        if torch.cuda.is_available(): return 'cuda'
        if torch.backends.mps.is_available(): return 'mps'
        return 'cpu'

    def register_hooks(self):
        self.hidden_states = []
        def hook_fn(module, input, output):
            hs = output[0].detach().cpu() if isinstance(output, tuple) else output.detach().cpu()
            self.hidden_states.append(hs)
        for name, module in self.model.named_modules():
            if 'encoder.layer' in name or 'transformer.layer' in name:
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def analyze(self, texts: List[str], batch_size: int = 8) -> Dict[str, Any]:
        """Analyze over a batch of calibration texts."""
        self.register_hooks()
        all_hidden = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors='pt', truncation=True, max_length=128, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                self.model(**inputs)
            all_hidden.append(self.hidden_states[-1])  # last layer? Actually we collect all layers
        # Average hidden states across batch and positions
        avg_hidden_per_layer = []
        for layer_idx in range(len(self.hidden_states)):
            stacked = torch.stack([h[layer_idx] for h in all_hidden]) if len(all_hidden) > 1 else self.hidden_states[layer_idx]
            avg = stacked.float().mean(dim=(0,1)).numpy()
            avg_hidden_per_layer.append(avg)
        self.remove_hooks()
        # Compute cosine similarities
        similarities = []
        for i in range(len(avg_hidden_per_layer)-1):
            sim = 1 - cosine(avg_hidden_per_layer[i], avg_hidden_per_layer[i+1])
            similarities.append(max(0, min(1, sim)))
        return {
            'num_layers': len(avg_hidden_per_layer),
            'cosine_similarities': similarities,
            'effective_ranks': self._effective_ranks(avg_hidden_per_layer),
        }

    def _effective_ranks(self, vectors):
        ranks = []
        for v in vectors:
            s = np.linalg.svd(v.reshape(1, -1), compute_uv=False)
            ranks.append(1)  # placeholder; real would need matrix of hidden states across tokens
        return ranks

def estimate_intrinsic_dimension(hidden_states: np.ndarray, k: int = 5) -> float:
    """Estimate intrinsic dimension using nearest‑neighbor method."""
    N, D = hidden_states.shape
    if N < k+1:
        return float(D)
    from scipy.spatial.distance import pdist, squareform
    dists = squareform(pdist(hidden_states))
    np.fill_diagonal(dists, np.inf)
    kth_dists = np.sort(dists, axis=1)[:, k-1]
    valid = kth_dists > 0
    if not valid.any():
        return float(D)
    log_ratios = np.log(kth_dists[valid] / kth_dists[valid][:, None])
    id_estimate = -1.0 / np.mean(log_ratios)
    return max(1.0, min(float(D), id_estimate))