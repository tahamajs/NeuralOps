"""
Compute embeddings for research interests using sentence-transformers.
"""

import logging

import numpy as np
from django.conf import settings

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)

# Load model once
_model = None


def get_model():
    global _model
    if SentenceTransformer is None:
        return None
    if _model is None:
        model_name = getattr(settings, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        logger.info("Loading embedding model: %s", model_name)
        try:
            _model = SentenceTransformer(model_name)
        except Exception as e:
            logger.warning("Could not load sentence transformer: %s", e)
            _model = None
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

    model = get_model()
    if model is None:
        return None

    texts = []
    for paper in papers:
        # Combine title + abstract for richer context
        text = f"{paper.title or ''} {paper.abstract or ''}"[:1000]
        texts.append(text)

    try:
        embeddings = model.encode(texts, show_progress_bar=False)
    except Exception as e:
        logger.warning("Embedding encode failed: %s", e)
        return None
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
