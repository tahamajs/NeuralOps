"""
Keyword router – rule-based routing using pattern matching and keywords.

Routes tasks based on input text patterns and predefined rules.
"""

# framework/routing/keyword_router.py
import re
from collections import defaultdict
from typing import Dict, Optional, Tuple

from .base import BaseRouter
from .router_registry import ROUTER_REGISTRY

DEFAULT_KEYWORDS: Dict[str, Dict[str, float]] = {
    "comparative": {
        "better": 1.2,
        "worse": 1.2,
        "than": 0.8,
        "more": 0.6,
        "less": 0.6,
        "higher": 1.0,
        "lower": 1.0,
        "superior": 1.5,
        "inferior": 1.5,
        "best": 1.3,
        "worst": 1.3,
        "prefer": 1.2,
        "instead": 0.8,
        "compared": 1.0,
        "versus": 1.0,
        "vs": 1.0,
        "outperform": 1.4,
        "surpass": 1.4,
        "exceed": 1.2,
        "greater": 1.0,
        "smaller": 1.0,
        "faster": 1.0,
        "slower": 1.0,
        "cheaper": 1.0,
        "expensive": 1.0,
    },
    "negation": {
        "not": 1.0,
        "no": 1.0,
        "never": 1.2,
        "nobody": 1.0,
        "nothing": 1.0,
        "neither": 1.0,
        "nor": 0.8,
        "without": 0.8,
        "n't": 1.0,
        "don't": 1.0,
        "won't": 1.0,
        "can't": 1.0,
        "shouldn't": 1.0,
        "wouldn't": 1.0,
        "couldn't": 1.0,
        "didn't": 1.0,
        "hasn't": 1.0,
        "isn't": 1.0,
        "aren't": 1.0,
        "wasn't": 1.0,
        "weren't": 1.0,
        "hardly": 0.8,
        "barely": 0.8,
        "scarcely": 0.8,
        "rarely": 0.6,
    },
    "sentiment_positive": {
        "good": 1.0,
        "great": 1.2,
        "awesome": 1.5,
        "amazing": 1.5,
        "wonderful": 1.3,
        "excellent": 1.4,
        "fantastic": 1.5,
        "love": 1.5,
        "like": 0.8,
        "enjoy": 1.0,
        "happy": 1.2,
        "satisfied": 1.0,
        "pleased": 1.0,
        "perfect": 1.3,
    },
    "sentiment_negative": {
        "terrible": 1.5,
        "awful": 1.5,
        "horrible": 1.5,
        "bad": 1.2,
        "poor": 1.0,
        "hate": 1.5,
        "dislike": 1.0,
        "sad": 1.2,
        "angry": 1.0,
        "frustrated": 1.0,
        "disappointed": 1.2,
    },
    "question": {
        "what": 1.0,
        "why": 1.0,
        "how": 1.0,
        "when": 1.0,
        "where": 1.0,
        "who": 1.0,
        "which": 1.0,
        "whom": 0.8,
        "does": 0.8,
        "do": 0.6,
        "is": 0.6,
        "are": 0.6,
        "was": 0.6,
        "were": 0.6,
        "can": 0.8,
        "could": 0.8,
        "would": 0.8,
        "should": 0.8,
        "?": 2.0,
    },
    "factual": {
        "according": 1.2,
        "research": 1.2,
        "study": 1.2,
        "shows": 1.0,
        "demonstrates": 1.0,
        "evidence": 1.3,
        "data": 1.0,
        "statistics": 1.2,
        "results": 0.8,
        "findings": 1.0,
        "published": 1.0,
        "report": 0.8,
        "analysis": 1.0,
        "survey": 0.8,
    },
    "opinion": {
        "think": 1.0,
        "believe": 1.0,
        "feel": 1.0,
        "opinion": 1.2,
        "view": 0.8,
        "perspective": 0.8,
        "personally": 1.2,
        "in my opinion": 1.5,
        "i think": 1.5,
        "i believe": 1.5,
    },
}


@ROUTER_REGISTRY.register("keyword_router")
class KeywordRouter(BaseRouter):
    name = "keyword_router"

    def __init__(
        self,
        keywords: Optional[Dict[str, Dict[str, float]]] = None,
        case_sensitive: bool = False,
        use_word_boundaries: bool = True,
        min_confidence: float = 0.3,
        default_task: str = "factual",
    ) -> None:
        self.keywords = keywords or DEFAULT_KEYWORDS
        self.case_sensitive = case_sensitive
        self.use_word_boundaries = use_word_boundaries
        self.min_confidence = min_confidence
        self.default_task = default_task
        self._patterns = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """ compile patterns."""
        for task, word_weights in self.keywords.items():
            if self.use_word_boundaries:
                pattern = r"\b(" + "|".join(re.escape(w) for w in word_weights) + r")\b"
            else:
                pattern = "|".join(re.escape(w) for w in word_weights)
            flags = 0 if self.case_sensitive else re.IGNORECASE
            self._patterns[task] = re.compile(pattern, flags)

    def route(self, text: str) -> Tuple[str, Dict[str, float]]:
        """Route."""
        scores = defaultdict(float)
        text_proc = text if self.case_sensitive else text.lower()
        for task, pattern in self._patterns.items():
            matches = pattern.findall(text_proc)
            word_weights = self.keywords[task]
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1] if len(match) > 1 else ""
                scores[task] += word_weights.get(match, 1.0)
        if not scores:
            return self.default_task, {self.default_task: 1.0}
        total = sum(scores.values())
        best = max(scores, key=scores.get)
        confidence = scores[best] / total
        if confidence < self.min_confidence:
            return self.default_task, {self.default_task: 1.0}
        probs = {k: v / total for k, v in scores.items()}
        return best, probs

    def can_route(self, text: str) -> bool:
        """Can route."""
        return True

    def add_keywords(self, task: str, word_weights: Dict[str, float]) -> None:
        """Add keywords."""
        if task not in self.keywords:
            self.keywords[task] = {}
        self.keywords[task].update(word_weights)
        self._compile_patterns()
