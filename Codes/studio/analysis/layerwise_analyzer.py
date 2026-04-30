# studio/analysis/layerwise_analyzer.py
import torch
import numpy as np
from scipy.spatial.distance import cosine

class LayerwiseAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_states = []
    def register_hooks(self):
        def hook(module, input, output, name):
            self.hidden_states.append(output.detach().cpu())
        for name, mod in self.model.named_modules():
            if 'layer' in name:
                mod.register_forward_hook(lambda m, i, o, n=name: hook(m, i, o, n))
    def compute_cosine_similarities(self):
        sims = []
        for i in range(len(self.hidden_states)-1):
            h1 = self.hidden_states[i].mean(dim=(0,1)).numpy()
            h2 = self.hidden_states[i+1].mean(dim=(0,1)).numpy()
            sims.append(1 - cosine(h1, h2))
        return sims
    def analyze(self, text):
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        self.hidden_states = []
        self.register_hooks()
        with torch.no_grad():
            self.model(**inputs)
        return {'cosine_similarities': self.compute_cosine_similarities()}