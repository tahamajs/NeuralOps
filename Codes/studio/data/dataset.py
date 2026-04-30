# studio/data/dataset.py
import torch
import json
import pandas as pd
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(text, truncation=True, padding='max_length',
                             max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class MultiChoiceDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.samples = []
        for _, row in df.iterrows():
            q = str(row.get('question', row.get('prompt', '')))
            choices = []
            if 'choices_json' in row:
                try:
                    choices = json.loads(row['choices_json'])
                except: pass
            elif 'endings' in row:
                try:
                    choices = json.loads(row['endings'])
                except: pass
            else:
                ch = [row.get(f'choice_{chr(ord("a")+i)}', '') for i in range(4)]
                choices = [c for c in ch if c]
            label = int(row.get('label', 0))
            if len(choices) >= 2:
                self.samples.append((q, choices, label))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        q, choices, label = self.samples[idx]
        encs = []
        for c in choices:
            enc = self.tokenizer(f"{q} {c}", truncation=True, padding='max_length',
                                 max_length=128, return_tensors='pt')
            encs.append({'input_ids': enc['input_ids'].squeeze(0),
                         'attention_mask': enc['attention_mask'].squeeze(0)})
        input_ids = torch.stack([e['input_ids'] for e in encs])
        attention_mask = torch.stack([e['attention_mask'] for e in encs])
        return {'input_ids': input_ids, 'attention_mask': attention_mask,
                'labels': torch.tensor(label, dtype=torch.long)}