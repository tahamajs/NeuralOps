"""CompressionTrainer and metric builders."""

import numpy as np
from transformers import Trainer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr

class CompressionTrainer(Trainer):
    """Trainer that adds orthogonal loss (if any) to task loss."""
    def __init__(self, ortho_weight: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ortho_weight = ortho_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        task_loss = outputs.loss
        ortho_loss = model.get_ortho_loss()
        total_loss = task_loss + self.ortho_weight * ortho_loss
        return (total_loss, outputs) if return_outputs else total_loss

def get_compute_metrics(task_name: str):
    """Return a compute_metrics function for a GLUE task."""
    t = task_name.lower()

    if t == "cola":
        def compute(p):
            preds = np.argmax(p.predictions, axis=-1)
            return {"eval_matthews_correlation": matthews_corrcoef(p.label_ids, preds)}
        return compute
    if t == "stsb":
        def compute(p):
            preds = p.predictions.squeeze()
            corr, _ = pearsonr(p.label_ids, preds)
            return {"eval_pearson": corr}
        return compute
    if t in ("mrpc", "qqp"):
        def compute(p):
            preds = np.argmax(p.predictions, axis=-1)
            return {"eval_accuracy": accuracy_score(p.label_ids, preds),
                    "eval_f1": f1_score(p.label_ids, preds)}
        return compute
    # default accuracy
    def compute(p):
        preds = np.argmax(p.predictions, axis=-1)
        return {"eval_accuracy": accuracy_score(p.label_ids, preds)}
    return compute
