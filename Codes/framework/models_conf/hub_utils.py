"""Search and list HuggingFace Hub models."""

import threading
import time

from huggingface_hub import HfApi

_cache_lock = threading.Lock()
_cache = {}


def _cache_get(key):
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (time.time() - entry[0]) < 300:
            return entry[1]
    return None


def _cache_set(key, value):
    with _cache_lock:
        _cache[key] = (time.time(), value)


POPULAR_MODELS = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
    "albert-base-v2",
    "google/electra-small-discriminator",
    "microsoft/deberta-v3-base",
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
    "google/gemma-2b",
    "facebook/opt-125m",
    "t5-small",
    "gpt2",
]


def search_models(query: str = "", limit: int = 20) -> list:
    """
    Search HuggingFace Hub for models matching `query`.
    Returns a list of model IDs (strings).
    Uses a simple cache with a 5-minute expiry.
    """
    if not query.strip():
        return POPULAR_MODELS[:limit]
    key = f"hf_search_{query}_{limit}"
    cached = _cache_get(key)
    if cached:
        return cached
    try:
        api = HfApi()
        models = api.list_models(search=query, limit=limit)
        results = [m.modelId for m in models]
        _cache_set(key, results)
        return results
    except Exception:
        import logging

        logging.getLogger(__name__).warning(
            "HF Hub search failed, returning empty list"
        )
        return []


def get_model_info(model_name: str) -> dict:
    """
    Fetch basic metadata about a HuggingFace model.
    Returns a dict with 'id', 'pipeline_tag', 'tags', or just the model name on error.
    """
    try:
        api = HfApi()
        info = api.model_info(model_name)
        return {
            "id": info.modelId,
            "pipeline_tag": info.pipeline_tag,
            "tags": info.tags,
        }
    except Exception:
        return {"id": model_name}
