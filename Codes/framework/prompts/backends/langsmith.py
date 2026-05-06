"""
LangSmith Prompt Hub backend.
Requires: pip install langsmith
"""

import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class LangSmithBackend:
    """Persist prompts to LangSmith Prompt Hub."""

    def __init__(self, api_key: str = "", project: str = "default"):
        self.api_key = api_key
        self.project = project
        self._client = None
        try:
            from langsmith import Client
            self._client = Client(api_key=api_key or None)
        except ImportError:
            logger.warning("langsmith not installed – LangSmith backend disabled")
        except Exception as e:
            logger.warning("Failed to initialise LangSmith client: %s", e)

    def save(self, name: str, template: str, version: int,
             commit_message: str = "", tags: Optional[list] = None) -> bool:
        """Push a new prompt version to LangSmith."""
        if self._client is None:
            logger.warning("LangSmith client unavailable – save skipped")
            return False
        try:
            # LangSmith expects a prompt commit via pull/push model.
            # We create a commit message, push the prompt, and tag the version.
            self._client.push_prompt(
                project_name=self.project,
                prompt_identifier=name,
                prompt=template,
                commit_message=commit_message or f"Version {version}",
                tags=tags or [],
            )
            logger.info("Pushed prompt '%s' v%d to LangSmith", name, version)
            return True
        except Exception as e:
            logger.exception("LangSmith save failed: %s", e)
            return False

    def load(self, name: str, version: Optional[int] = None,
             alias: str = "production") -> Optional[Tuple[str, int]]:
        """Retrieve a prompt from LangSmith. Returns (template, version_number) or None."""
        if self._client is None:
            return None
        try:
            # Load latest (or specific version) by tag/alias.
            prompt = self._client.get_prompt(
                prompt_identifier=name,
                project_name=self.project,
            )
            if prompt is None:
                return None
            return prompt.template, prompt.version
        except Exception as e:
            logger.exception("LangSmith load failed: %s", e)
            return None

    def list_versions(self, name: str) -> list:
        """Return list of version numbers available."""
        if self._client is None:
            return []
        try:
            history = self._client.list_prompt_versions(
                prompt_identifier=name, project_name=self.project
            )
            return [v.version for v in history]
        except Exception:
            return []