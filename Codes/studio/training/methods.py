# studio/training/methods.py
"""
Unified training pipeline supporting:
- baseline
- lora
- dora
- rdrop
- r3f
- mixout
- llrd
+ combinations (e.g., lora+rdrop)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from typing import Dict, Optional, Any

from ..compression.svd_linear import SVDLinearLayer
from ..attentions.utils import replace_attention_in_model


# ---------- LoRA / DoRA / Mixout / LLRD ----------
def apply_lora(model: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.1):
    from peft import LoraConfig, get_peft_model, TaskType
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["query", "key", "value", "dense"]
    )
    return get_peft_model(model, config)

def apply_dora(model: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.1):
    # DoRA implementation from your framework (already in framework/attentions/dora.py)
    from ..attentions.dora import DoRALinear
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(x in name for x in ['query','key','value','dense']):
            parent = model.get_submodule('.'.join(name.split('.')[:-1])) if '.' in name else model
            child = name.split('.')[-1]
            setattr(parent, child, DoRALinear(module, r, alpha, dropout))
    return model

def apply_mixout(model: nn.Module, p: float = 0.5):
    from ..compression.mixout import MixoutLinear
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'classifier' not in name:
            parent = model.get_submodule('.'.join(name.split('.')[:-1])) if '.' in name else model
            child = name.split('.')[-1]
            setattr(parent, child, MixoutLinear(module, p))
    return model

def get_llrd_param_groups(model: nn.Module, base_lr: float, decay: float = 0.95, weight_decay: float = 0.01):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    layers = []
    for name, param in model.named_parameters():
        if "encoder.layer." in name:
            layer = int(name.split("encoder.layer.")[1].split(".")[0])
            layers.append((name, param, layer))
    max_layer = max(l[2] for l in layers) if layers else 0
    groups = []
    # embeddings
    emb_params = [p for n,p in model.named_parameters() if 'embeddings' in n]
    if emb_params:
        groups.append({'params': emb_params, 'lr': base_lr * (decay ** (max_layer+1)), 'weight_decay': weight_decay})
    # layers
    for l in range(max_layer+1):
        lr = base_lr * (decay ** (max_layer - l))
        decay_params = []
        nodecay_params = []
        for n,p in model.named_parameters():
            if f"encoder.layer.{l}." in n:
                if any(nd in n for nd in no_decay):
                    nodecay_params.append(p)
                else:
                    decay_params.append(p)
        if decay_params:
            groups.append({'params': decay_params, 'lr': lr, 'weight_decay': weight_decay})
        if nodecay_params:
            groups.append({'params': nodecay_params, 'lr': lr, 'weight_decay': 0.0})
    # head
    head_params = [p for n,p in model.named_parameters() if 'classifier' in n or 'pooler' in n]
    if head_params:
        groups.append({'params': head_params, 'lr': base_lr, 'weight_decay': weight_decay})
    return groups


# ---------- R‑Drop loss ----------
def rdrop_loss(logits1, logits2, labels, is_regression=False, alpha=5.0):
    if is_regression:
        loss1 = F.mse_loss(logits1.view(-1), labels.view(-1))
        loss2 = F.mse_loss(logits2.view(-1), labels.view(-1))
        kl = 0.5 * (F.kl_div(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1), reduction='batchmean') +
                    F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1), reduction='batchmean'))
        return (loss1 + loss2)/2 + alpha * kl
    else:
        ce = nn.CrossEntropyLoss()
        loss1 = ce(logits1, labels)
        loss2 = ce(logits2, labels)
        kl = 0.5 * (F.kl_div(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1), reduction='batchmean') +
                    F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1), reduction='batchmean'))
        return (loss1 + loss2)/2 + alpha * kl


# ---------- Custom Trainer with method support ----------
class MethodTrainer(Trainer):
    def __init__(self, method: str, rdrop_alpha: float = 5.0, r3f_epsilon: float = 1e-5, is_regression: bool = False, **kwargs):
        super().__init__(**kwargs)
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
            # simplified: add noise to embeddings
            emb = model.get_input_embeddings()
            inputs_embeds = emb(inputs["input_ids"])
            noise = torch.randn_like(inputs_embeds) * self.r3f_epsilon
            noisy_inputs = dict(inputs)
            noisy_inputs["inputs_embeds"] = inputs_embeds + noise
            outputs_clean = model(**inputs)
            outputs_noisy = model(**noisy_inputs)
            loss = rdrop_loss(outputs_clean.logits, outputs_noisy.logits, inputs["labels"],
                              self.is_regression, 1.0)  # KL weight = 1
            return (loss, outputs_clean) if return_outputs else loss
        return super().compute_loss(model, inputs, return_outputs, **kwargs)


def get_trainer_for_method(method: str, model, tokenizer, train_dataset, eval_dataset, output_dir: str, **kwargs):
    """
    Factory to create a Trainer with the correct method.
    """
    from transformers import TrainingArguments
    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=kwargs.get("learning_rate", 2e-5),
        per_device_train_batch_size=kwargs.get("batch_size", 8),
        per_device_eval_batch_size=kwargs.get("eval_batch_size", 16),
        num_train_epochs=kwargs.get("epochs", 3),
        weight_decay=kwargs.get("weight_decay", 0.01),
        warmup_ratio=kwargs.get("warmup_ratio", 0.1),
        max_grad_norm=kwargs.get("max_grad_norm", 1.0),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model=kwargs.get("metric", "eval_accuracy"),
        greater_is_better=True,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
    )
    # Apply method-specific modifications to model
    if "lora" in method and "lora+rdrop" not in method:
        model = apply_lora(model, r=kwargs.get("lora_r", 8), alpha=kwargs.get("lora_alpha", 16))
    if "dora" in method:
        model = apply_dora(model, r=kwargs.get("lora_r", 8), alpha=kwargs.get("lora_alpha", 16))
    if "mixout" in method:
        model = apply_mixout(model, p=kwargs.get("mixout_p", 0.5))
    if "llrd" in method:
        param_groups = get_llrd_param_groups(model, args.learning_rate, decay=kwargs.get("llrd_decay", 0.95))
        from torch.optim import AdamW
        optimizer = AdamW(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = None  # will use default

    trainer = MethodTrainer(
        method=method,
        rdrop_alpha=kwargs.get("rdrop_alpha", 5.0),
        r3f_epsilon=kwargs.get("r3f_epsilon", 1e-5),
        is_regression=kwargs.get("is_regression", False),
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        optimizers=(optimizer, None) if optimizer else None,
        compute_metrics=kwargs.get("compute_metrics"),
    )
    return trainer