"""
NLP Adversarial Robustness Testing – inspired by TextAttack, the industry standard.

Supports character‑level, word‑level, and sentence‑level perturbations with
automatic evaluation of model robustness.

Reference: https://github.com/QData/TextAttack
"""

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from framework.core.registry import Registry

ROBUSTNESS_REGISTRY = Registry("robustness_attacks")


@dataclass
class RobustnessReport:
    """Complete adversarial robustness analysis."""

    baseline_accuracy: float
    attack_results: List[Dict[str, Any]]
    overall_robustness_score: float  # 0 = fragile, 1 = robust
    successful_attacks: int
    total_samples_tested: int
    worst_attack: str
    recommendations: List[str] = field(default_factory=list)


class TextAdversarialTester:
    """
    Test NLP model robustness against adversarial perturbations.

    Usage:
        tester = TextAdversarialTester()
        def predict_fn(texts: List[str]) -> np.ndarray:
            # return model predictions
            ...
        report = tester.evaluate(predict_fn, texts, labels)
    """

    def __init__(self, max_perturbation_ratio: float = 0.3, seed: int = 42) -> None:
        self.max_perturbation_ratio = max_perturbation_ratio
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    # ── Main API ────────────────────────────────────────────────────
    def evaluate(
        self,
        predict_fn: Callable[[List[str]], np.ndarray],
        texts: List[str],
        labels: np.ndarray,
        attacks: Optional[List[str]] = None,
        max_samples: int = 200,
    ) -> RobustnessReport:
        """Run all requested attacks and return a robustness report."""
        if attacks is None:
            attacks = [
                "char_swap",
                "char_delete",
                "word_substitute",
                "word_delete",
                "word_swap",
                "add_typo",
            ]

        # Use at most max_samples
        if len(texts) > max_samples:
            indices = np.random.choice(len(texts), max_samples, replace=False)
            texts = [texts[i] for i in indices]
            labels = labels[indices]

        # Baseline
        baseline_preds = predict_fn(texts)
        baseline_acc = np.mean(baseline_preds == labels)

        all_results = []
        for attack_name in attacks:
            result = self._run_attack(
                attack_name, predict_fn, texts, labels, baseline_preds
            )
            all_results.append(result)

        # Compute overall robustness score
        success_rates = [r.get("success_rate", 0) for r in all_results]
        overall = 1.0 - np.mean(success_rates) if success_rates else 1.0

        worst = (
            max(all_results, key=lambda r: r.get("success_rate", 0))
            if all_results
            else {}
        )
        recs = self._generate_recommendations(all_results)

        return RobustnessReport(
            baseline_accuracy=baseline_acc,
            attack_results=all_results,
            overall_robustness_score=round(overall, 4),
            successful_attacks=sum(r.get("n_flipped", 0) for r in all_results),
            total_samples_tested=len(texts) * len(attacks),
            worst_attack=worst.get("attack", ""),
            recommendations=recs,
        )

    def _run_attack(
        self, name: str, predict_fn, texts, labels, baseline_preds
    ) -> Dict[str, Any]:
        """Run a single attack type."""
        perturbed = [self._perturb(t, name) for t in texts]
        preds = predict_fn(perturbed)
        flipped = preds != baseline_preds
        n_flipped = int(np.sum(flipped))
        success_rate = n_flipped / len(texts)

        # Average confidence drop on flipped samples
        avg_perturbations = np.mean(
            [self._perturbation_distance(o, p) for o, p in zip(texts, perturbed)]
        )

        return {
            "attack": name,
            "n_flipped": n_flipped,
            "success_rate": round(success_rate, 4),
            "avg_num_perturbations": round(avg_perturbations, 1),
        }

    # ── Perturbation methods ────────────────────────────────────────
    def _perturb(self, text: str, method: str) -> str:
        """Apply a single perturbation to a text."""
        if method == "char_swap":
            return self._char_swap(text)
        elif method == "char_delete":
            return self._char_delete(text)
        elif method == "word_substitute":
            return self._word_substitute(text)
        elif method == "word_delete":
            return self._word_delete(text)
        elif method == "word_swap":
            return self._word_swap(text)
        elif method == "add_typo":
            return self._add_typo(text)
        elif method == "insert_punctuation":
            return self._insert_punctuation(text)
        elif method == "repeat_char":
            return self._repeat_char(text)
        return text

    def _char_swap(self, text: str) -> str:
        """ char swap."""
        chars = list(text)
        n = max(1, int(len(chars) * self.max_perturbation_ratio * 0.3))
        for _ in range(n):
            if len(chars) >= 2:
                i = random.randint(0, len(chars) - 2)
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)

    def _char_delete(self, text: str) -> str:
        """ char delete."""
        chars = list(text)
        n = max(1, int(len(chars) * self.max_perturbation_ratio * 0.3))
        for _ in range(n):
            if len(chars) > 1:
                idx = random.randint(0, len(chars) - 1)
                chars.pop(idx)
        return "".join(chars)

    def _word_substitute(self, text: str) -> str:
        """ word substitute."""
        words = text.split()
        n = max(1, int(len(words) * self.max_perturbation_ratio))
        for _ in range(n):
            if len(words) > 0:
                idx = random.randint(0, len(words) - 1)
                # Simple substitution: replace with a common word
                words[idx] = random.choice(
                    [
                        "the",
                        "a",
                        "is",
                        "are",
                        "was",
                        "were",
                        "very",
                        "really",
                        "some",
                        "any",
                    ]
                )
        return " ".join(words)

    def _word_delete(self, text: str) -> str:
        """ word delete."""
        words = text.split()
        n = max(1, int(len(words) * self.max_perturbation_ratio))
        indices = set(random.sample(range(len(words)), min(n, len(words))))
        return " ".join(w for i, w in enumerate(words) if i not in indices)

    def _word_swap(self, text: str) -> str:
        """ word swap."""
        words = text.split()
        n = max(1, int(len(words) * self.max_perturbation_ratio * 0.3))
        for _ in range(n):
            if len(words) >= 2:
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
        return " ".join(words)

    def _add_typo(self, text: str) -> str:
        """Replace characters with keyboard‑adjacent keys."""
        keyboard_adjacent = {
            "a": "qwsz",
            "b": "vghn",
            "c": "xdfv",
            "d": "serfcx",
            "e": "wsdfr",
            "t": "rfghy",
            "s": "awedxz",
            "l": "kop",
        }
        chars = list(text)
        n = max(1, int(len(chars) * self.max_perturbation_ratio * 0.3))
        for _ in range(n):
            idx = random.randint(0, len(chars) - 1)
            c = chars[idx].lower()
            if c in keyboard_adjacent:
                chars[idx] = random.choice(keyboard_adjacent[c])
        return "".join(chars)

    def _insert_punctuation(self, text: str) -> str:
        """ insert punctuation."""
        words = text.split()
        n = max(1, int(len(words) * self.max_perturbation_ratio * 0.3))
        for _ in range(n):
            idx = random.randint(0, len(words) - 1)
            words[idx] += random.choice(".,!?;:")
        return " ".join(words)

    def _repeat_char(self, text: str) -> str:
        """ repeat char."""
        chars = list(text)
        n = max(1, int(len(chars) * self.max_perturbation_ratio * 0.3))
        for _ in range(n):
            idx = random.randint(0, len(chars) - 1)
            chars.insert(idx, chars[idx])
        return "".join(chars)

    def _perturbation_distance(self, original: str, perturbed: str) -> float:
        """Levenshtein‑like distance normalised by length."""
        if len(original) == 0:
            return float(len(perturbed))
        # Simple character difference count (approximation)
        shorter = min(len(original), len(perturbed))
        diffs = sum(1 for i in range(shorter) if original[i] != perturbed[i])
        diffs += abs(len(original) - len(perturbed))
        return diffs / max(len(original), 1)

    def _generate_recommendations(self, results: List[Dict]) -> List[str]:
        """ generate recommendations."""
        recs = []
        for r in results:
            if r.get("success_rate", 0) > 0.3:
                recs.append(
                    f"Model is vulnerable to '{r['attack']}' attacks "
                    f"(success rate: {r['success_rate']:.1%}). "
                    f"Consider adversarial training with this perturbation type."
                )
        if not recs:
            recs.append("Model is robust against all tested attack types.")
        return recs
