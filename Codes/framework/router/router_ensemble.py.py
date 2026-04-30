# framework/routing/ml_router.py
"""
Machine learning-based task routing using fast text classifiers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


class MLRouter:
    """
    ML-based router using TF-IDF + Logistic Regression.
    
    Example:
        >>> router = MLRouter()
        >>> router.train(train_texts, train_labels)
        >>> task = router.route("This is a comparative sentence")
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, multi_class='auto'))
        ])
        
        if model_path:
            self.load(model_path)
    
    def train(self, texts: List[str], labels: List[str]) -> 'MLRouter':
        """
        Train the router on labeled data.
        
        Args:
            texts: List of input texts
            labels: List of task labels
            
        Returns:
            Self for chaining
        """
        self.model.fit(texts, labels)
        return self
    
    def route(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Route text to a task type.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (predicted_task, probabilities)
        """
        probs = self.model.predict_proba([text])[0]
        pred = self.model.classes_[np.argmax(probs)]
        return pred, dict(zip(self.model.classes_, probs))
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump(self.model, path)
    
    def load(self, path: str):
        """Load model from disk."""
        self.model = joblib.load(path)


class FastTextRouter(nn.Module):
    """
    FastText-style neural router for task detection.
    
    Architecture: Embedding -> Average -> FFN -> Softmax
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 7,
        max_seq_len: int = 128
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.max_seq_len = max_seq_len
    
    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, emb_dim)
        # Average pooling
        pooled = emb.mean(dim=1)  # (batch, emb_dim)
        pooled = self.dropout(F.relu(self.fc1(pooled)))
        logits = self.fc2(pooled)
        return logits
    
    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)