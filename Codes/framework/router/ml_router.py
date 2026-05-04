"""
Machine learning routers (TF‑IDF + LR, FastText neural).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import Dict, List, Optional, Tuple
import joblib
from .base import TrainableRouter


class TfidfRouter(TrainableRouter):
    """
    TF‑IDF + Logistic Regression router.
    Fast, interpretable, works well for small to medium datasets.
    """
    
    name = "tfidf_router"
    
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        **lr_kwargs
    ):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
            ('clf', LogisticRegression(max_iter=1000, multi_class='auto', **lr_kwargs))
        ])
        self.classes_: Optional[List[str]] = None
        self.is_trained = False
    
    def train(self, texts: List[str], labels: List[str], **kwargs) -> 'TfidfRouter':
        self.pipeline.fit(texts, labels)
        self.classes_ = self.pipeline.classes_
        self.is_trained = True
        return self
    
    def route(self, text: str) -> Tuple[str, Dict[str, float]]:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        probs = self.pipeline.predict_proba([text])[0]
        pred = self.pipeline.classes_[np.argmax(probs)]
        return pred, dict(zip(self.pipeline.classes_, probs))
    
    def can_route(self, text: str) -> bool:
        return self.is_trained and len(text.strip()) > 0
    
    def save(self, path: str):
        joblib.dump(self.pipeline, path)
    
    def load(self, path: str):
        self.pipeline = joblib.load(path)
        self.classes_ = self.pipeline.classes_
        self.is_trained = True


class FastTextRouter(TrainableRouter, nn.Module):
    """
    Neural router inspired by FastText: embedding averaging + FFN.
    Supports GPU training and batching.
    """
    
    name = "fasttext_router"
    
    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 7,
        max_seq_len: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.classes_: Optional[List[str]] = None
        self.is_trained = False
        self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        emb = self.embedding(x)                     # (B, L, E)
        pooled = emb.mean(dim=1)                    # (B, E)
        pooled = self.dropout(torch.relu(self.fc1(pooled)))
        logits = self.fc2(pooled)
        return logits
    
    def train_model(
        self,
        texts: List[str],
        labels: List[str],
        tokenizer: callable,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.001,
        validation_split: float = 0.1
    ) -> Dict[str, List[float]]:
        """
        Train the neural router.
        
        Args:
            texts: List of input text strings
            labels: List of string labels
            tokenizer: Function that converts a string to a list of token ids
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            validation_split: Fraction for validation
            
        Returns:
            Training history dict
        """
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, TensorDataset
        import torch.optim as optim
        
        # Map labels to indices
        unique_labels = sorted(set(labels))
        self.classes_ = unique_labels
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
        # Adjust output layer
        self.fc2 = nn.Linear(self.fc2.in_features, self.num_classes).to(self.embedding.weight.device)
        
        # Tokenize
        token_ids = []
        for t in texts:
            ids = tokenizer(t)[:self.max_seq_len]
            if len(ids) < self.max_seq_len:
                ids = ids + [0] * (self.max_seq_len - len(ids))
            token_ids.append(ids)
        X = torch.tensor(token_ids, dtype=torch.long)
        y = torch.tensor([label_to_idx[lbl] for lbl in labels], dtype=torch.long)
        
        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            correct = 0
            total = 0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.embedding.weight.device), yb.to(self.embedding.weight.device)
                optimizer.zero_grad()
                out = self.forward(Xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, pred = torch.max(out, 1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
            
            val_loss, val_acc = self._evaluate(val_loader, criterion)
            train_acc = correct / total
            avg_loss = total_loss / len(train_loader)
            history['loss'].append(avg_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
        
        self.is_trained = True
        return history
    
    def _evaluate(self, loader, criterion):
        self.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.embedding.weight.device), yb.to(self.embedding.weight.device)
                out = self.forward(Xb)
                loss = criterion(out, yb)
                total_loss += loss.item()
                _, pred = torch.max(out, 1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        return total_loss / len(loader), correct / total
    
    def route(self, text: str) -> Tuple[str, Dict[str, float]]:
        if not self.is_trained or not hasattr(self, 'tokenizer'):
            raise RuntimeError("Model not trained. Call train_model() first.")
        ids = self.tokenizer(text)[:self.max_seq_len]
        if len(ids) < self.max_seq_len:
            ids = ids + [0] * (self.max_seq_len - len(ids))
        X = torch.tensor([ids], dtype=torch.long, device=self.embedding.weight.device)
        with torch.no_grad():
            logits = self.forward(X)[0]
            probs = F.softmax(logits, dim=0).cpu().numpy()
        pred_idx = np.argmax(probs)
        pred_label = self.classes_[pred_idx]
        scores = {label: probs[i] for i, label in enumerate(self.classes_)}
        return pred_label, scores
    
    def can_route(self, text: str) -> bool:
        return self.is_trained and hasattr(self, 'tokenizer')
    
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'classes': self.classes_,
            'max_seq_len': self.max_seq_len,
        }, path)
    
    def load(self, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.classes_ = checkpoint['classes']
        self.max_seq_len = checkpoint['max_seq_len']
        self.num_classes = len(self.classes_)
        self.is_trained = True