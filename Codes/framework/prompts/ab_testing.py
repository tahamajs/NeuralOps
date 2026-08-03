"""A/B testing with traffic splitting and gradual rollout."""

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ABTestConfig:
    """Configuration for an A/B test on a prompt."""

    prompt_name: str
    variants: Dict[str, float]  # alias → traffic percentage (must total 100)
    metric: str = "accuracy"
    min_sample_size: int = 100
    auto_promote: bool = False  # automatically promote winner to production
    auto_promote_threshold: float = 0.95  # confidence threshold for auto‑promote


@dataclass
class ABTestResult:
    variant: str
    requests: int
    avg_score: float
    confidence_interval: tuple
    is_winner: bool = False


class ABTestRouter:
    """
    Traffic splitting router for A/B testing prompts.

    Modeled after: PromptLayer Dynamic Release Labels, LangSmith Environments,
    and Literal AI progressive rollouts.

    Usage:
        router = ABTestRouter()
        router.create_test("customer_support", {"v1": 80.0, "v2": 20.0})
        alias = router.route("customer_support")   # returns "v1" 80% of the time
    """

    def __init__(self) -> None:
        self._tests: Dict[str, ABTestConfig] = {}
        self._counters: Dict[str, Dict[str, int]] = {}  # prompt → {variant → requests}
        self._scores: Dict[str, Dict[str, List[float]]] = {}

    def create_test(
        self,
        prompt_name: str,
        variants: Dict[str, float],
        metric: str = "accuracy",
        **kwargs,
    ) -> ABTestConfig:
        """Set up a new A/B test. Variants is a dict of alias → percentage."""
        total = sum(variants.values())
        if abs(total - 100.0) > 0.01:
            raise ValueError(f"Variant percentages must total 100, got {total}")
        cfg = ABTestConfig(
            prompt_name=prompt_name, variants=variants, metric=metric, **kwargs
        )
        self._tests[prompt_name] = cfg
        self._counters[prompt_name] = {v: 0 for v in variants}
        self._scores[prompt_name] = {v: [] for v in variants}
        logger.info("A/B test created for '%s': %s", prompt_name, variants)
        return cfg

    def route(self, prompt_name: str) -> str:
        """Return the alias (variant) to use for this request."""
        cfg = self._tests.get(prompt_name)
        if cfg is None:
            return "production"  # fallback
        r = random.uniform(0, 100)
        cumulative = 0.0
        for variant, pct in cfg.variants.items():
            cumulative += pct
            if r <= cumulative:
                self._counters[prompt_name][variant] += 1
                return variant
        return list(cfg.variants.keys())[-1]  # fallback

    def record_result(self, prompt_name: str, variant: str, score: float) -> None:
        """Record an outcome for statistical analysis."""
        if prompt_name in self._scores:
            self._scores[prompt_name].setdefault(variant, []).append(score)

    def get_results(self, prompt_name: str) -> List[ABTestResult]:
        """Compute summary statistics for each variant."""
        cfg = self._tests.get(prompt_name)
        if cfg is None:
            return []
        import numpy as np

        results = []
        best_score, best_variant = -1, None
        for variant in cfg.variants:
            scores = self._scores.get(prompt_name, {}).get(variant, [])
            requests = self._counters.get(prompt_name, {}).get(variant, 0)
            if len(scores) < 2:
                results.append(
                    ABTestResult(
                        variant=variant,
                        requests=requests,
                        avg_score=0,
                        confidence_interval=(0, 0),
                    )
                )
                continue
            mean = np.mean(scores)
            se = np.std(scores, ddof=1) / np.sqrt(len(scores))
            ci = (mean - 1.96 * se, mean + 1.96 * se)
            results.append(
                ABTestResult(
                    variant=variant,
                    requests=requests,
                    avg_score=mean,
                    confidence_interval=ci,
                )
            )
            if mean > best_score:
                best_score, best_variant = mean, variant
        if best_variant:
            for r in results:
                if r.variant == best_variant:
                    r.is_winner = True
        return results

    def promote_winner(self, prompt_name: str, registry) -> Optional[str]:
        """Promote the winning variant to production."""
        results = self.get_results(prompt_name)
        winners = [r for r in results if r.is_winner]
        if not winners:
            return None
        winner = winners[0].variant
        registry.set_alias(
            prompt_name, "production", registry.get_alias(prompt_name, winner)
        )
        logger.info("A/B test winner promoted: '%s' → %s", prompt_name, winner)
        return winner
