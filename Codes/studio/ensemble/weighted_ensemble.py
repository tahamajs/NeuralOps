# studio/ensemble/weighted_ensemble.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TaskAwareAttention(nn.Module):
    def __init__(self, num_tasks, hidden_dim=64):
        super().__init__()
        self.task_emb = nn.Embedding(num_tasks, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.sim_mat = nn.Parameter(torch.eye(num_tasks))
    def forward(self, task_id, all_logits, model_task_ids):
        device = all_logits.device
        cur_emb = self.q_proj(self.task_emb(torch.tensor([task_id], device=device)))
        task_embs = self.k_proj(self.task_emb(torch.tensor(model_task_ids, device=device)))
        scores = torch.matmul(cur_emb, task_embs.T) / np.sqrt(self.task_emb.embedding_dim)
        scores += self.sim_mat[model_task_ids, task_id].unsqueeze(0)
        weights = F.softmax(scores, dim=-1)
        weighted = (all_logits * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        return weighted

class SmartWeightedEnsemble:
    def __init__(self, models, num_tasks):
        self.models = models
        self.performance = torch.ones(len(models), num_tasks) * 0.5
        self.attention = TaskAwareAttention(num_tasks)
    def combine_logits(self, task_id, all_logits, input_ids, attention_mask, method="hybrid"):
        if method == "simple_average":
            return all_logits.mean(dim=0)
        elif method == "weighted":
            weights = self._get_weights(task_id, input_ids, attention_mask)
            return (all_logits * weights).sum(dim=0)
        elif method == "task_aware_attention":
            model_task_ids = [i // (len(self.models)//num_tasks) for i in range(len(self.models))]
            return self.attention(task_id, all_logits, model_task_ids)
        else:  # hybrid
            model_task_ids = [i // (len(self.models)//num_tasks) for i in range(len(self.models))]
            attn_weighted = self.attention(task_id, all_logits, model_task_ids)
            dyn_weighted = (all_logits * self._get_weights(task_id, input_ids, attention_mask)).sum(dim=0)
            return 0.6 * attn_weighted + 0.4 * dyn_weighted

    def _get_weights(self, task_id, input_ids, attention_mask):
        # dummy – implement real confidence scoring in full version
        return torch.ones(len(self.models), device=input_ids.device) / len(self.models)

def run_smart_ensemble(config, dataset_path):
    print("Smart ensemble not fully implemented – use CLI or import manually")