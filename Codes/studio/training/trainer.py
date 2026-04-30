# studio/training/trainer.py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

def train_model(model, train_loader, val_loader, device, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attn_mask).logits
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        # validation
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attn_mask = batch['attention_mask'].to(device)
                lbls = batch['labels']
                logits = model(input_ids, attention_mask=attn_mask).logits
                preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                truths.extend(lbls.numpy())
        acc = accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average='weighted')
        print(f"Epoch {epoch+1} - loss: {total_loss/len(train_loader):.4f} - val_acc: {acc:.4f} - f1: {f1:.4f}")
    return model

def fine_tune_compressed_model(model, tokenizer, train_dataset, val_dataset, epochs=3, lr=2e-5):
    from torch.utils.data import DataLoader
    from .optimizer import get_optimizer
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    optimizer = get_optimizer(model, lr)
    return train_model(model, train_loader, val_loader, model.device, optimizer, epochs=epochs)