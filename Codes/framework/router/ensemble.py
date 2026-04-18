import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from ..compression.svd_linear import SVDLinearLayer
from ..utils.device import get_device

class RouterEnsemble:
    def __init__(self, tasks_config, device=None):
        self.tasks_config = tasks_config  # list of dicts: {'name':, 'path':, 'rank':}
        self.models = {}
        self.device = device or get_device()

    def load_and_compress(self, base_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        for task in self.tasks_config:
            print(f"Loading model for {task['name']}...")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
            # Apply SVD compression
            from ..compression.svd_linear import SVDLinearLayer
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and any(p in name.lower() for p in ['attention', 'dense', 'output']):
                    try:
                        new_layer = SVDLinearLayer(module, task['rank'])
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        parent = model.get_submodule(parent_name) if parent_name else model
                        setattr(parent, child_name, new_layer)
                    except:
                        pass
            model.to(self.device)
            self.models[task['name']] = {'model': model, 'tokenizer': tokenizer}
        return self

    def predict(self, text: str):
        detected = detect_task_type(text)
        if detected not in self.models:
            detected = list(self.models.keys())[0]
        item = self.models[detected]
        enc = item['tokenizer'](text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        with torch.no_grad():
            logits = item['model'](input_ids=enc['input_ids'].to(self.device), attention_mask=enc['attention_mask'].to(self.device)).logits
            pred = torch.argmax(logits, dim=-1).item()
        return pred, detected

    def evaluate_on_full_dataset(self, data_files: dict):
        all_texts, all_labels, all_sources = [], [], []
        for task_name, path in data_files.items():
            df = pd.read_csv(path)
            tcol = [c for c in df.columns if 'text' in c.lower()][0]
            lcol = [c for c in df.columns if 'label' in c.lower()][0]
            df = df.rename(columns={tcol: 'text', lcol: 'label'})
            all_texts.extend(df['text'].tolist())
            all_labels.extend(df['label'].tolist())
            all_sources.extend([task_name] * len(df))
        preds = [self.predict(t)[0] for t in all_texts]
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds, average='weighted')
        return {'accuracy': acc, 'f1': f1}