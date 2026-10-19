#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Label quality detection – inspired by Cleanlab, the industry standard for
data‑centric AI.  Finds label errors, outliers, and near‑duplicates.

Reference: https://github.com/cleanlab/cleanlab
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neuralops.core.registry import Registry

LABEL_QUALITY_REGISTRY = Registry("label_quality_methods")


@dataclass
class LabelQualityReport:
    """Complete label quality analysis for a dataset."""

    n_samples: int
    n_label_issues: int
    issue_rate: float
    label_issue_indices: np.ndarray
    label_quality_scores: np.ndarray
    suggested_labels: np.ndarray | None = None
    per_class_issues: dict[int, int] = field(default_factory=dict)
    outlier_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    near_duplicate_pairs: list[tuple[int, int]] = field(default_factory=list)
    overall_quality_score: float = 0.0


class LabelQualityDetector:
    """
    Detects label errors, outliers, and near‑duplicates using confident learning.

    Usage:
        detector = LabelQualityDetector()
        report = detector.analyze(labels=y_true, pred_probs=model_output_probs)

    No ground‑truth validation set is required – the method works entirely from
    model predictions and given labels.
    """

    def __init__(self, threshold: float = 0.1, seed: int = 42) -> None:
        """Execute init routine."""
        self.threshold = threshold
        self.seed = seed

    def analyze(
        self,
        labels: np.ndarray,
        pred_probs: np.ndarray,
        class_names: list[str] | None = None,
    ) -> LabelQualityReport:
        """
        Run full label quality analysis.

        Args:
            labels: integer labels, shape (N,)
            pred_probs: model predicted probabilities, shape (N, K)
            class_names: optional list of class name strings
        """
        n_samples, n_classes = pred_probs.shape
        np.random.seed(self.seed)

        cj = self._compute_confident_joint(labels, pred_probs)

        label_issues_mask, quality_scores, suggested = self._find_label_issues(
            labels, pred_probs, cj
        )

        per_class = {}
        for k in range(n_classes):
            mask_k = labels == k
            per_class[k] = int(np.sum(label_issues_mask[mask_k]))

        entropy = -np.sum(pred_probs * np.log(pred_probs + 1e-10), axis=1)
        outlier_thresh = np.percentile(entropy, 95)
        outlier_indices = np.where(entropy > outlier_thresh)[0]

        near_dupes = self._find_near_duplicates(pred_probs)

        overall = 1.0 - np.mean(label_issues_mask)

        return LabelQualityReport(
            n_samples=n_samples,
            n_label_issues=int(np.sum(label_issues_mask)),
            issue_rate=round(np.mean(label_issues_mask), 4),
            label_issue_indices=np.where(label_issues_mask)[0],
            label_quality_scores=quality_scores,
            suggested_labels=suggested,
            per_class_issues=per_class,
            outlier_indices=outlier_indices,
            near_duplicate_pairs=near_dupes,
            overall_quality_score=round(overall, 4),
        )

    def _compute_confident_joint(self, labels: np.ndarray, pred_probs: np.ndarray) -> np.ndarray:
        """
        Estimate the joint distribution of noisy labels and true labels.
        This is the key insight from Cleanlab / confident learning.
        """
        n_samples, n_classes = pred_probs.shape
        pred_classes = np.argmax(pred_probs, axis=1)

        thresholds = np.zeros(n_classes)
        for k in range(n_classes):
            mask = pred_classes == k
            if mask.sum() > 0:
                thresholds[k] = np.mean(pred_probs[mask, k])
            else:
                thresholds[k] = 0.5

        cj = np.zeros((n_classes, n_classes), dtype=np.float64)
        for i in range(n_samples):
            true_label_guess = np.argmax(pred_probs[i])
            given_label = labels[i]
            if pred_probs[i, true_label_guess] >= thresholds[true_label_guess]:
                cj[given_label, true_label_guess] += 1

        cj = cj / max(cj.sum(), 1.0)
        return cj

    def _find_label_issues(
        self, labels: np.ndarray, pred_probs: np.ndarray, cj: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Return: (issue_mask, quality_scores, suggested_labels)
        """
        n_samples, n_classes = pred_probs.shape

        self_confidence = pred_probs[np.arange(n_samples), labels]
        quality_scores = self_confidence

        thresholds = np.zeros(n_classes)
        for k in range(n_classes):
            mask = labels == k
            if mask.sum() > 0:
                thresholds[k] = np.mean(pred_probs[mask, k])
            else:
                thresholds[k] = 0.5

        issue_mask = np.zeros(n_samples, dtype=bool)
        for k in range(n_classes):
            mask_k = labels == k
            issue_mask[mask_k] = self_confidence[mask_k] < (thresholds[k] * (1 - self.threshold))

        suggested = np.argmax(pred_probs, axis=1)
        suggested[~issue_mask] = labels[~issue_mask]

        return issue_mask, quality_scores, suggested

    def _find_near_duplicates(
        self, pred_probs: np.ndarray, threshold: float = 0.95
    ) -> list[tuple[int, int]]:
        """Find pairs of samples whose prediction vectors are nearly identical."""
        from sklearn.metrics.pairwise import cosine_similarity

        n = pred_probs.shape[0]
        if n > 5000:
            indices = np.random.choice(n, min(5000, n), replace=False)
            subset = pred_probs[indices]
            sim = cosine_similarity(subset)
        else:
            sim = cosine_similarity(pred_probs)
        np.fill_diagonal(sim, 0)
        pairs = np.argwhere(sim > threshold)
        seen = set()
        result = []
        for i, j in pairs:
            if i < j and (i, j) not in seen:
                result.append((int(i), int(j)))
                seen.add((i, j))
        return result

    def summary(self, report: LabelQualityReport) -> dict[str, Any]:
        """Summary."""
        return {
            "total_samples": report.n_samples,
            "label_issues_found": report.n_label_issues,
            "issue_rate": report.issue_rate,
            "overall_quality": report.overall_quality_score,
            "n_outliers": len(report.outlier_indices),
            "n_near_duplicates": len(report.near_duplicate_pairs),
            "worst_classes": sorted(
                report.per_class_issues.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }
