"""
ML Router – machine learning-based intelligent routing for task dispatch.

Dynamically routes tasks based on features and learned patterns.
"""

# framework/routing/ml_router.py
from typing import Any
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .base import TrainableRouter
from .router_registry import ROUTER_REGISTRY


@ROUTER_REGISTRY.register("tfidf_router")
class TfidfRouter(TrainableRouter):
    name = "tfidf_router"

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        **lr_kwargs
    ) -> None:
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(max_features=max_features, ngram_range=ngram_range),
                ),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, multi_class="auto", **lr_kwargs),
                ),
            ]
        )
        self.classes_: Optional[List[str]] = None
        self.is_trained = False

    def train(self, texts: List[str], labels: List[str], **kwargs) -> "TfidfRouter":
        """Fit the model on the provided dataset."""
        self.pipeline.fit(texts, labels)
        self.classes_ = self.pipeline.classes_
        self.is_trained = True
        return self

    def route(self, text: str) -> Tuple[str, Dict[str, float]]:
        """Route."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        probs = self.pipeline.predict_proba([text])[0]
        pred = self.pipeline.classes_[np.argmax(probs)]
        return pred, dict(zip(self.pipeline.classes_, probs))

    def can_route(self, text: str) -> bool:
        """Can route."""
        return self.is_trained and len(text.strip()) > 0

    def save(self, path: str) -> None:
        """Save."""
        joblib.dump(self.pipeline, path)

    def load(self, path: str) -> None:
        """Load."""
        self.pipeline = joblib.load(path)
        self.classes_ = self.pipeline.classes_
        self.is_trained = True


@ROUTER_REGISTRY.register("fasttext_router")
class FastTextRouter(TrainableRouter, nn.Module):
    name = "fasttext_router"

    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 7,
        max_seq_len: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.classes_: Optional[List[str]] = None
        self.is_trained = False
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self._device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — compute module output from inputs."""
        emb = self.embedding(x)
        pooled = emb.mean(dim=1)
        pooled = self.dropout(torch.relu(self.fc1(pooled)))
        return self.fc2(pooled)

    def train_model(
        self,
        texts: List[str],
        labels: List[str],
        tokenizer,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.001,
        validation_split: float = 0.1,
    ) -> Any:
        """Train model."""
        import torch.optim as optim
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, TensorDataset

        unique_labels = sorted(set(labels))
        self.classes_ = unique_labels
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
        self.fc2 = nn.Linear(self.fc2.in_features, self.num_classes).to(self._device)

        token_ids = []
        for t in texts:
            ids = tokenizer(t)[: self.max_seq_len]
            if len(ids) < self.max_seq_len:
                ids += [0] * (self.max_seq_len - len(ids))
            token_ids.append(ids)
        X = torch.tensor(token_ids, dtype=torch.long)
        y = torch.tensor([label_to_idx[lbl] for lbl in labels], dtype=torch.long)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

        for epoch in range(epochs):
            self.train()
            total_loss, correct, total = 0, 0, 0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self._device), yb.to(self._device)
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
            history["loss"].append(total_loss / len(train_loader))
            history["accuracy"].append(correct / total)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

        self.is_trained = True
        return history

    def _evaluate(self, loader, criterion) -> Any:
        """ evaluate."""
        self.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for Xb, yb in loader:
                Xb, yb = Xb.to(self._device), yb.to(self._device)
                out = self.forward(Xb)
                loss = criterion(out, yb)
                total_loss += loss.item()
                _, pred = torch.max(out, 1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        return total_loss / len(loader), correct / total

    def route(self, text: str) -> Tuple[str, Dict[str, float]]:
        """Route."""
        if not self.is_trained or not hasattr(self, "tokenizer"):
            raise RuntimeError("Model not trained or tokenizer missing")
        ids = self.tokenizer(text)[: self.max_seq_len]
        if len(ids) < self.max_seq_len:
            ids += [0] * (self.max_seq_len - len(ids))
        X = torch.tensor([ids], dtype=torch.long, device=self._device)
        with torch.no_grad():
            logits = self.forward(X)[0]
            probs = F.softmax(logits, dim=0).cpu().numpy()
        pred_idx = np.argmax(probs)
        pred_label = self.classes_[pred_idx]
        scores = {label: probs[i] for i, label in enumerate(self.classes_)}
        return pred_label, scores

    def can_route(self, text: str) -> bool:
        """Can route."""
        return self.is_trained and hasattr(self, "tokenizer")

    def save(self, path: str) -> None:
        """Save."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "classes": self.classes_,
                "max_seq_len": self.max_seq_len,
            },
            path,
        )

    def load(self, path: str, map_location=None) -> None:
        """Load."""
        # FIX: Enforce weights_only=False only when loading custom class attributes or use safe deserialization; here weights_only=False with trusted source or weights_only=True for state_dict
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.classes_ = checkpoint["classes"]
        self.max_seq_len = checkpoint["max_seq_len"]
        self.num_classes = len(self.classes_)
        self.is_trained = True
