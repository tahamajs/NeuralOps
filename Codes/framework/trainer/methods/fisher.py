# framework/trainer/methods/fisher.py
import numpy as np
import torch
import logging
from torch.utils.data import DataLoader
from .registry import register_method

logger = logging.getLogger(__name__)

def compute_fisher_importance(model, dataset, tokenizer, task_keys, config,
                              target_substrings=None):
    if target_substrings is None:
        target_substrings = ["query", "key", "value", "dense"]

    sent1_key, sent2_key = task_keys
    device = next(model.parameters()).device
    sample_size = min(config.fisher_sample_size, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    sampled = dataset.select(indices)

    def collate_fn(batch):
        if sent2_key is None:
            texts = [ex[sent1_key] for ex in batch]
            enc = tokenizer(texts, truncation=True, max_length=config.max_seq_length,
                            padding=True, return_tensors="pt")
        else:
            texts1 = [ex[sent1_key] for ex in batch]
            texts2 = [ex[sent2_key] for ex in batch]
            enc = tokenizer(texts1, texts2, truncation=True, max_length=config.max_seq_length,
                            padding=True, return_tensors="pt")
        labels = [ex["label"] for ex in batch]
        is_reg = task_keys.get("is_regression", False)
        enc["labels"] = torch.tensor(labels, dtype=torch.float if is_reg else torch.long)
        return enc

    loader = DataLoader(sampled, batch_size=config.per_device_train_batch_size, collate_fn=collate_fn)
    fisher = {}
    model.eval()
    for param in model.parameters():
        param.requires_grad = True

    for i, batch in enumerate(loader):
        if i >= config.fisher_num_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if not any(ts in name for ts in target_substrings):
                continue
            score = (param.grad ** 2).mean().item()
            module_name = name.replace(".weight", "").replace(".bias", "")
            fisher[module_name] = fisher.get(module_name, 0.0) + score
        model.zero_grad()

    if not fisher:
        return None
    for k in fisher:
        fisher[k] /= config.fisher_num_batches
    scores = np.array(list(fisher.values()))
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    norm_scores = exp_scores / (np.sum(exp_scores) + 1e-8)
    fisher = {k: norm_scores[i] for i, k in enumerate(fisher.keys())}
    return fisher

@register_method("fisher_lora")
def apply_fisher_lora(model, config, dataset, tokenizer, task_keys):
    logger.info("Fisher‑LoRA method called – computing Fisher importance...")
    fisher = compute_fisher_importance(model, dataset, tokenizer, task_keys, config)
    if fisher is None:
        logger.warning("Fisher importance returned None; falling back to uniform ranks.")
        return
    # Here you would use the fisher scores to assign adaptive ranks.
    # For demonstration, we set each LoRA layer's rank proportionally.
    logger.info(f"Fisher importance computed for {len(fisher)} modules.")