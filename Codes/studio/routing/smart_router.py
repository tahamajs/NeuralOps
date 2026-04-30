# studio/routing/smart_router.py
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .keyword_router import detect_task_type
from ..compression.svd_linear import SVDLinearLayer

class SmartRouter:
    def __init__(self, models_config, device):
        self.models = {}
        self.device = device
        for cfg in models_config:
            name = cfg["name"]
            tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
            model = AutoModelForSequenceClassification.from_pretrained(cfg["path"], num_labels=2).to(device)
            # apply SVD
            for n, m in model.named_modules():
                if isinstance(m, torch.nn.Linear) and any(p in n for p in ['attention', 'q_lin', 'k_lin', 'v_lin']):
                    setattr(model.get_submodule('.'.join(n.split('.')[:-1])), n.split('.')[-1],
                            SVDLinearLayer(m, cfg.get("rank", 32)))
            self.models[name] = (model, tokenizer)

    def predict(self, text):
        task = detect_task_type(text)
        if task not in self.models:
            task = list(self.models.keys())[0]
        model, tokenizer = self.models[task]
        enc = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            logits = model(**enc).logits
            pred = torch.argmax(logits, dim=-1).item()
        return pred, task

def run_smart_router(config, data_dir="./data"):
    from ..config import config as global_config
    from ..data.loader import discover_datasets
    tasks = discover_datasets(data_dir)[:3]
    models_cfg = [{"name": t["name"], "path": global_config.model.name, "rank": 32} for t in tasks]
    router = SmartRouter(models_cfg, global_config.get_device())
    # evaluate on first dataset
    ds = tasks[0]
    df = pd.read_csv(ds["path"])
    text_col = [c for c in df.columns if 'text' in c.lower()][0]
    labels = df['label'] if 'label' in df.columns else df.iloc[:,-1]
    correct = 0
    for i, row in df.iterrows():
        pred, _ = router.predict(row[text_col])
        if pred == int(labels.iloc[i]):
            correct += 1
    acc = correct / len(df)
    print(f"Router accuracy on {ds['name']}: {acc:.4f}")