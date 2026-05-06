import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RobustnessReport:
    method: str; baseline_accuracy: float
    perturbed_accuracies: List[float]; perturbations: List[float]
    accuracy_drop_at_max: float; is_robust: bool; threshold: float

class RobustnessTester:
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def test(self, model, tokenizer, texts: List[str], labels: np.ndarray,
             perturbation_levels: List[float] = None,
             methods: List[str] = None) -> List[RobustnessReport]:
        if perturbation_levels is None: perturbation_levels = [0.0, 0.01, 0.05, 0.1]
        if methods is None: methods = ["token_drop", "token_swap", "typo"]
        reports = []
        baseline_acc = self._evaluate(model, tokenizer, texts, labels)
        for method in methods:
            accs = [baseline_acc]
            for eps in perturbation_levels[1:]:
                perturbed = self._perturb(texts, method, eps)
                acc = self._evaluate(model, tokenizer, perturbed, labels)
                accs.append(acc)
            max_drop = baseline_acc - min(accs)
            reports.append(RobustnessReport(
                method=method, baseline_accuracy=baseline_acc,
                perturbed_accuracies=accs, perturbations=perturbation_levels,
                accuracy_drop_at_max=max_drop,
                is_robust=max_drop <= self.threshold, threshold=self.threshold
            ))
        return reports

    def _perturb(self, texts: List[str], method: str, eps: float) -> List[str]:
        import random
        result = []
        for text in texts:
            words = text.split()
            n = max(1, int(len(words) * eps))
            if method == "token_drop":
                indices = random.sample(range(len(words)), min(n, len(words)))
                result.append(" ".join(w for i, w in enumerate(words) if i not in indices))
            elif method == "token_swap":
                for _ in range(n):
                    if len(words) >= 2:
                        i, j = random.sample(range(len(words)), 2)
                        words[i], words[j] = words[j], words[i]
                result.append(" ".join(words))
            elif method == "typo":
                chars = list(text)
                for _ in range(min(n, len(chars))):
                    idx = random.randint(0, len(chars)-1)
                    chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz")
                result.append("".join(chars))
        return result

    def _evaluate(self, model, tokenizer, texts, labels):
        import torch
        model.eval()
        correct = 0
        with torch.no_grad():
            for i, text in enumerate(texts):
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
                outputs = model(**{k: v.to(model.device) for k, v in inputs.items()})
                pred = outputs.logits.argmax(dim=1).item()
                if pred == labels[i]: correct += 1
        return correct / len(texts) if texts else 0