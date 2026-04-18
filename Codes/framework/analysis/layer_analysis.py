import torch
import numpy as np
from scipy.spatial.distance import cosine
import json

class LayerwiseAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_states = []
        self.hooks = []

    def register_hooks(self):
        def hook_fn(module, input, output, name):
            self.hidden_states.append(output.detach().cpu())
        for name, module in self.model.named_modules():
            if 'encoder.layer' in name:
                hook = module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
                self.hooks.append(hook)

    def compute_cosine_similarities(self):
        sims = []
        for i in range(len(self.hidden_states)-1):
            h1 = self.hidden_states[i].mean(dim=(0,1)).numpy()
            h2 = self.hidden_states[i+1].mean(dim=(0,1)).numpy()
            sims.append(1 - cosine(h1, h2))
        return sims

    def compute_effective_ranks(self):
        ranks = []
        for hs in self.hidden_states:
            H = hs.reshape(-1, hs.shape[-1]).numpy()
            s = np.linalg.svd(H, compute_uv=False)
            s_norm = s / s.sum()
            entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
            ranks.append(np.exp(entropy))
        return ranks

    def analyze(self, text):
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        self.hidden_states = []
        self.register_hooks()
        with torch.no_grad():
            self.model(**inputs)
        self._remove_hooks()
        return {
            'cosine_similarities': self.compute_cosine_similarities(),
            'effective_ranks': self.compute_effective_ranks(),
        }

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []