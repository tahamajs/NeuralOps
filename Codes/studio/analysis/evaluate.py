# studio/analysis/evaluate.py
import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

def evaluate_model(model, dataloader, device):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            logits = model(input_ids, attention_mask=attn_mask).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            truths.extend(labels.numpy())
    return {
        'accuracy': accuracy_score(truths, preds),
        'f1_weighted': f1_score(truths, preds, average='weighted')
    }

def benchmark_model(model, dataloader, device):
    model.eval()
    times = []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        start = time.time()
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attn_mask)
        if device.type == 'cuda': torch.cuda.synchronize()
        times.append(time.time() - start)
    return {'avg_batch_time': np.mean(times)}