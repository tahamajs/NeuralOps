"""Per‑version evaluation tracking for prompts."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class EvalMetric:
    name: str
    value: float
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptEvaluation:
    """
    Tracks evaluation metrics for a specific prompt version.
    Records: accuracy, latency, token counts, cost, user feedback, etc.

    Modeled after: PromptForge, Datadog LLM Observability, LangSmith evaluations.
    """

    prompt_name: str
    version: int
    # Standard metrics
    requests: int = 0
    accuracy: Optional[float] = None
    consistency: Optional[float] = None
    relevance: Optional[float] = None
    coherence: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    total_tokens: int = 0
    avg_tokens_per_request: float = 0.0
    total_cost_usd: float = 0.0
    avg_cost_per_request: float = 0.0
    user_feedback_positive: int = 0
    user_feedback_negative: int = 0
    # Custom metrics
    custom_metrics: List[EvalMetric] = field(default_factory=list)
    last_evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset_name: Optional[str] = None
    evaluator_model: Optional[str] = None

    @property
    def feedback_ratio(self) -> float:
        total = self.user_feedback_positive + self.user_feedback_negative
        return self.user_feedback_positive / total if total > 0 else 0.0

    def log_request(
        self, latency_ms: float, tokens: int, cost: float, feedback: int = 0
    ):
        """Log a single request's statistics."""
        self.requests += 1
        self.avg_latency_ms = (
            (self.avg_latency_ms or 0) * (self.requests - 1) + latency_ms
        ) / self.requests
        self.total_tokens += tokens
        self.avg_tokens_per_request = self.total_tokens / self.requests
        self.total_cost_usd += cost
        self.avg_cost_per_request = self.total_cost_usd / self.requests
        if feedback > 0:
            self.user_feedback_positive += 1
        elif feedback < 0:
            self.user_feedback_negative += 1
        self.last_evaluated_at = datetime.now().isoformat()

    def add_custom_metric(self, name: str, value: float, unit: str = ""):
        self.custom_metrics.append(EvalMetric(name=name, value=value, unit=unit))

    def to_dict(self) -> dict:
        return {
            "prompt_name": self.prompt_name,
            "version": self.version,
            "requests": self.requests,
            "accuracy": self.accuracy,
            "consistency": self.consistency,
            "relevance": self.relevance,
            "coherence": self.coherence,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_request": self.avg_tokens_per_request,
            "total_cost_usd": self.total_cost_usd,
            "avg_cost_per_request": self.avg_cost_per_request,
            "user_feedback_positive": self.user_feedback_positive,
            "user_feedback_negative": self.user_feedback_negative,
            "feedback_ratio": self.feedback_ratio,
            "custom_metrics": [
                {"name": m.name, "value": m.value, "unit": m.unit}
                for m in self.custom_metrics
            ],
            "last_evaluated_at": self.last_evaluated_at,
            "dataset_name": self.dataset_name,
            "evaluator_model": self.evaluator_model,
        }
