"""
Complete Prompt Management System – modeled after LangSmith, MLflow Prompt Registry,
Qwak, PromptLayer, and Langfuse.

Features:
  - Centralised Prompt Registry with CRUD operations
  - Git‑like versioning with auto‑increment, commit messages, and diff
  - Aliases / Environments (production, staging, development)
  - A/B testing with traffic splitting and gradual rollout
  - Template engine (simple strings + chat conversation templates)
  - Per‑version evaluation tracking (latency, tokens, cost, scores)
  - Tags & metadata for organisation
  - Pluggable storage backends (local JSON, MLflow, LangSmith)
  - Hot‑loading for dynamic reload without code changes
  - Rollback with environment assignment history
"""

from .ab_testing import ABTestConfig, ABTestRouter
from .config import PromptConfig, PromptType, StorageBackend
from .evaluation import EvalMetric, PromptEvaluation
from .registry import PromptRegistry
from .templates import ChatPromptTemplate, PromptTemplate
from .versioning import PromptVersion, VersionHistory

__all__ = [
    "PromptConfig",
    "PromptType",
    "StorageBackend",
    "PromptRegistry",
    "PromptTemplate",
    "ChatPromptTemplate",
    "PromptVersion",
    "VersionHistory",
    "ABTestConfig",
    "ABTestRouter",
    "PromptEvaluation",
    "EvalMetric",
]
