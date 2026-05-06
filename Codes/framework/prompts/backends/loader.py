"""Hot‑loading prompt loader for dynamic reload without code changes."""

import threading, time, logging
from typing import Optional, Union
from .registry import PromptRegistry
from .templates import PromptTemplate, ChatPromptTemplate

logger = logging.getLogger(__name__)

class PromptLoader:
    """
    Wraps a PromptRegistry and supports hot‑loading (periodic refresh).

    Usage:
        loader = PromptLoader(registry, hot_load=True, refresh_interval=30)
        prompt = loader.get("customer_support", alias="production")
    """

    def __init__(self, registry: PromptRegistry,
                 hot_load: bool = False, refresh_interval: int = 60):
        self.registry = registry
        self.hot_load = hot_load
        self.refresh_interval = refresh_interval
        self._cache: dict = {}
        self._stop = threading.Event()
        if hot_load:
            self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
            self._thread.start()

    def get(self, name: str, version: int = None,
            alias: str = "production") -> Union[PromptTemplate, ChatPromptTemplate]:
        return self.registry.load(name, version=version, alias=alias)

    def _refresh_loop(self):
        while not self._stop.is_set():
            time.sleep(self.refresh_interval)
            try:
                # Reload from disk / backend
                self.registry._prompts.clear()
                self.registry._load_from_disk()
                logger.debug("Prompt hot‑load refresh complete")
            except Exception as e:
                logger.error("Hot‑load refresh failed: %s", e)

    def stop(self):
        self._stop.set()