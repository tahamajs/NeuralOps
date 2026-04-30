# studio/ensemble/robust_ensemble.py
import torch
import torch.nn as nn

class RobustEnsembleModel(nn.Module):
    def __init__(self, base_models, num_labels=2):
        super().__init__()
        self.models = nn.ModuleList(base_models)
        self.num_labels = num_labels
        self.weights = nn.Parameter(torch.ones(len(base_models), num_labels))
    def forward(self, inputs):
        logits_list = []
        for model in self.models:
            out = model(**inputs).logits
            if out.shape[-1] != self.num_labels:
                if out.shape[-1] > self.num_labels:
                    out = out[:, :self.num_labels]
                else:
                    pad = torch.zeros(out.shape[0], self.num_labels - out.shape[-1], device=out.device)
                    out = torch.cat([out, pad], dim=-1)
            logits_list.append(out)
        stacked = torch.stack(logits_list, dim=0)
        w = self.weights.unsqueeze(1)
        return (stacked * w).sum(dim=0)