"""
R‑Drop and R3F (regularised dropout / robust representation learning).
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

from .registry import register_method

logger = logging.getLogger(__name__)


def rdrop_loss(logits1, logits2, labels, is_regression, alpha) -> Any:
    """Rdrop loss."""
    if is_regression:
        loss_fn = nn.MSELoss()
        loss1 = loss_fn(logits1.view(-1), labels.view(-1))
        loss2 = loss_fn(logits2.view(-1), labels.view(-1))
        p = torch.distributions.Normal(logits1, torch.ones_like(logits1))
        q = torch.distributions.Normal(logits2, torch.ones_like(logits2))
        kl = (
            torch.distributions.kl_divergence(p, q).mean()
            + torch.distributions.kl_divergence(q, p).mean()
        ) / 2.0
        loss = (loss1 + loss2) / 2.0 + alpha * kl
    else:
        ce = nn.CrossEntropyLoss()
        loss1 = ce(logits1.view(-1, logits1.size(-1)), labels.view(-1))
        loss2 = ce(logits2.view(-1, logits2.size(-1)), labels.view(-1))
        logp1 = F.log_softmax(logits1, dim=-1)
        logp2 = F.log_softmax(logits2, dim=-1)
        p1 = torch.softmax(logits1, dim=-1)
        p2 = torch.softmax(logits2, dim=-1)
        kl1 = F.kl_div(logp1, p2, reduction="batchmean")
        kl2 = F.kl_div(logp2, p1, reduction="batchmean")
        kl = (kl1 + kl2) / 2.0
        loss = (loss1 + loss2) / 2.0 + alpha * kl
    return loss


def r3f_forward(
    model, input_ids, attention_mask, token_type_ids, labels, epsilon, is_regression
) -> Any:
    """R3f forward."""
    base_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if token_type_ids is not None:
        base_kwargs["token_type_ids"] = token_type_ids
    out_clean = model(**base_kwargs)
    logits_clean = out_clean.logits

    embed = model.get_input_embeddings()
    inputs_embeds = embed(input_ids)
    noise = torch.randn_like(inputs_embeds) * epsilon
    noise_kwargs = {
        "inputs_embeds": inputs_embeds + noise,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if token_type_ids is not None:
        noise_kwargs["token_type_ids"] = token_type_ids
    out_noisy = model(**noise_kwargs)
    logits_noisy = out_noisy.logits

    if is_regression:
        mse = nn.MSELoss()
        loss_clean = mse(logits_clean.view(-1), labels.view(-1))
        loss_noisy = mse(logits_noisy.view(-1), labels.view(-1))
        p = torch.distributions.Normal(logits_clean, torch.ones_like(logits_clean))
        q = torch.distributions.Normal(logits_noisy, torch.ones_like(logits_noisy))
        kl = torch.distributions.kl_divergence(p, q).mean()
        loss = (loss_clean + loss_noisy) / 2.0 + kl
    else:
        ce = nn.CrossEntropyLoss()
        loss_clean = ce(logits_clean.view(-1, logits_clean.size(-1)), labels.view(-1))
        loss_noisy = ce(logits_noisy.view(-1, logits_noisy.size(-1)), labels.view(-1))
        logp_clean = F.log_softmax(logits_clean, dim=-1)
        logp_noisy = F.log_softmax(logits_noisy, dim=-1)
        p = torch.softmax(logits_clean, dim=-1)
        q = torch.softmax(logits_noisy, dim=-1)
        kl1 = F.kl_div(logp_clean, q, reduction="batchmean")
        kl2 = F.kl_div(logp_noisy, p, reduction="batchmean")
        loss = (loss_clean + loss_noisy) / 2.0 + (kl1 + kl2) / 2.0
    return loss, logits_clean


class MethodTrainer(Trainer):
    def __init__(self, method, rdrop_alpha, r3f_epsilon, is_regression, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.method = method
        self.rdrop_alpha = rdrop_alpha
        self.r3f_epsilon = r3f_epsilon
        self.is_regression = is_regression

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs) -> Any:
        """Compute and return loss."""
        if "rdrop" in self.method.lower():
            outputs1 = model(**inputs)
            outputs2 = model(**inputs)
            loss = rdrop_loss(
                outputs1.logits,
                outputs2.logits,
                inputs["labels"],
                self.is_regression,
                self.rdrop_alpha,
            )
            return (loss, outputs1) if return_outputs else loss
        elif "r3f" in self.method.lower():
            loss, logits = r3f_forward(
                model,
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                token_type_ids=inputs.get("token_type_ids"),
                labels=inputs["labels"],
                epsilon=self.r3f_epsilon,
                is_regression=self.is_regression,
            )
            return (loss, logits) if return_outputs else loss
        return super().compute_loss(model, inputs, return_outputs, **kwargs)


@register_method("rdrop")
def apply_rdrop(model, config) -> None:
    """Apply rdrop."""
    pass


@register_method("r3f")
def apply_r3f(model, config) -> None:
    """Apply r3f."""
    pass
