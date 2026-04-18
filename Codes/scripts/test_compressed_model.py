import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from framework.compression.svd_linear import SVDLinearLayer
from framework.utils.device import get_device
from config import config

class SimpleDataset:
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0), 'attention_mask': enc['attention_mask'].squeeze(0), 'labels': torch.tensor(self.labels[idx])}

def load_compressed_model(model_name, rank):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'attention' in name.lower():
            try:
                new_layer = SVDLinearLayer(module, rank)
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name, new_layer)
            except:
                pass
    return model, tokenizer

def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            labels.extend(batch['labels'].numpy())
    return accuracy_score(labels, preds)

def main():
    device = get_device()
    df = pd.read_csv("./data/combined_master_dataset.csv")
    if 'text' not in df.columns or 'label' not in df.columns:
        text_col = [c for c in df.columns if 'text' in c.lower()][0]
        label_col = [c for c in df.columns if 'label' in c.lower()][0]
        df = df.rename(columns={text_col: 'text', label_col: 'label'})
    model, tokenizer = load_compressed_model("distilbert-base-uncased-finetuned-sst-2-english", rank=32)
    model.to(device)
    dataset = SimpleDataset(df['text'].tolist(), df['label'].tolist(), tokenizer)
    loader = DataLoader(dataset, batch_size=32)
    acc = evaluate(model, loader, device)
    print(f"Compressed model accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()