"""
Compute embeddings for research interests using sentence-transformers.
"""

import logging

import numpy as np
from django.conf import settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Load model once
_model = None


def get_model():
    global _model
    if _model is None:
        model_name = getattr(settings, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        logger.info("Loading embedding model: %s", model_name)
        _model = SentenceTransformer(model_name)
    return _model


def generate_user_embedding(user):
    """
    Generate an embedding vector for a user based on their saved papers.
    Returns a list of floats.
    """
    from agents.models import ResearchPaper

    papers = ResearchPaper.objects.filter(saved_by=user)
    if not papers.exists():
        return None

    texts = []
    for paper in papers:
        # Combine title + abstract for richer context
        text = f"{paper.title or ''} {paper.abstract or ''}"[:1000]
        texts.append(text)

    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    # Average the paper embeddings
    avg_embedding = np.mean(embeddings, axis=0).tolist()
    return avg_embedding


def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embedding lists."""
    a = np.array(emb1)
    b = np.array(emb2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
