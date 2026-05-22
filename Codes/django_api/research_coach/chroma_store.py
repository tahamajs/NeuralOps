"""
ChromaDB vector store for experiment memory.
Stores embeddings of experiment summaries and enables semantic search.
"""
import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from django.conf import settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_client = None
_collection = None
_ef = None


def get_client():
    global _client, _collection, _ef
    if _client is None:
        persist_dir = getattr(settings, 'CHROMA_PERSIST_DIR', './chroma_data')
        _client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        _ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=getattr(settings, 'EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        )
        _collection = _client.get_or_create_collection(
            name="experiment_memory",
            embedding_function=_ef,
            metadata={"hnsw:space": "cosine"}
        )
    return _client, _collection


def add_experiment_memory(run_id, summary_text, metadata=None):
    """Add an experiment to the vector store."""
    _, collection = get_client()
    collection.add(
        documents=[summary_text],
        metadatas=[metadata or {}],
        ids=[str(run_id)]
    )
    logger.info("Added experiment %s to ChromaDB", run_id)


def search_similar_experiments(query_text, n_results=5):
    """Search for similar experiments by semantic similarity."""
    _, collection = get_client()
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results


def delete_experiment_memory(run_id):
    """Remove an experiment from the vector store."""
    _, collection = get_client()
    collection.delete(ids=[str(run_id)])