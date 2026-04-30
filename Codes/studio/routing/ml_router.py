# studio/routing/ml_router.py
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

class MLRouter:
    def __init__(self):
        self.pipeline = None
    def train(self, texts, labels):
        self.pipeline = LogisticRegression(max_iter=1000)
        vec = TfidfVectorizer()
        X = vec.fit_transform(texts)
        self.pipeline.fit(X, labels)
        self.vectorizer = vec
    def route(self, text):
        X = self.vectorizer.transform([text])
        proba = self.pipeline.predict_proba(X)[0]
        pred = self.pipeline.classes_[proba.argmax()]
        return pred, dict(zip(self.pipeline.classes_, proba))

class FastTextRouter(nn.Module):
    def __init__(self, vocab_size=50000, emb_dim=128, hidden=64, num_classes=3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.fc1 = nn.Linear(emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
    def forward(self, x):
        x = self.emb(x).mean(dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)