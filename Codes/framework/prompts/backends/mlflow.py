import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MLflowBackend:
    """Store prompts in the MLflow Prompt Registry."""

    def __init__(self, tracking_uri: str = "") -> None:
        self.tracking_uri = tracking_uri
        try:
            import mlflow

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
        except ImportError:
            logger.warning("MLflow not installed – MLflow backend disabled")

    def save(self, name: str, template: str, version: int, commit_message: str) -> None:
        """Save."""
        try:
            import mlflow

            mlflow.genai.register_prompt(
                name=name, template=template, commit_message=commit_message
            )
            if version > 0:
                mlflow.genai.set_prompt_alias(
                    name=name, alias=f"v{version}", version=version
                )
        except Exception as e:
            logger.exception("MLflow save failed: %s", e)

    def load(self, name: str, alias: str = "production") -> Optional[str]:
        """Load."""
        try:
            import mlflow

            prompt = mlflow.genai.load_prompt(name_or_uri=f"prompts:/{name}@{alias}")
            return prompt.template if hasattr(prompt, "template") else str(prompt)
        except Exception:
            return None
