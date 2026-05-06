import logging
logger = logging.getLogger(__name__)

class ModelRegistry:
    _entries = []

    @classmethod
    def register(cls, artifact_path: str, metrics: dict, tags: list = None, **kwargs):
        entry = {
            "artifact_path": artifact_path,
            "metrics": metrics,
            "tags": tags or [],
            "registered_at": str(datetime.now()),
        }
        cls._entries.append(entry)
        logger.info("Model registered: %s", entry)
        return entry