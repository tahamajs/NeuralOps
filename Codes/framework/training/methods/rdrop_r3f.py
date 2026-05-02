import torch, torch.nn as nn, torch.nn.functional as F
from transformers import Trainer
from .registry import register_method
import logging

logger = logging.getLogger(__name__)

def rdrop_loss(logits1, logits2, labels, is_regression, alpha):
    # ... (legacy)

def r3f_forward(model, input_ids, attention_mask, token_type_ids, labels, epsilon, is_regression):
    # ... (legacy)

class MethodTrainer(Trainer):
    def __init__(self, method, rdrop_alpha, r3f_epsilon, is_regression, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.rdrop_alpha = rdrop_alpha
        self.r3f_epsilon = r3f_epsilon
        self.is_regression = is_regression

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if "rdrop" in self.method:
            outputs1 = model(**inputs)
            outputs2 = model(**inputs)
            loss = rdrop_loss(outputs1.logits, outputs2.logits, inputs["labels"],
                              self.is_regression, self.rdrop_alpha)
            return (loss, outputs1) if return_outputs else loss
        elif "r3f" in self.method:
            loss, logits = r3f_forward(
                model, inputs["input_ids"], inputs.get("attention_mask"),
                inputs.get("token_type_ids"), inputs["labels"],
                self.r3f_epsilon, self.is_regression)
            return (loss, logits) if return_outputs else loss
        return super().compute_loss(model, inputs, return_outputs)

@register_method("rdrop")
def apply_rdrop(model, config):
    pass  # No model modification needed; trainer will handle

@register_method("r3f")
def apply_r3f(model, config):
    pass