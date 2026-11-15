#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for evaluation metrics and sklearn parity."""

from __future__ import annotations

import numpy as np
import pytest
import sklearn.metrics as sk_metrics
import torch

from neuralops.trainer.metrics import (
    accuracy_score,
    aggregate_results,
    compute_accuracy,
    compute_f1,
    compute_matthews_corrcoef,
    compute_pearson_spearman,
    compute_perplexity,
    compute_precision_recall,
    compute_regression_metrics,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def test_accuracy_score_exact_sklearn_parity():
    """Verify accuracy_score matches sklearn.metrics.accuracy_score on 1D arrays."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 1, 0, 2, 2])
    y_pred = np.array([0, 2, 2, 0, 1, 1, 1, 0, 2, 0])

    our_acc = accuracy_score(y_true, y_pred)
    sk_acc = sk_metrics.accuracy_score(y_true, y_pred)
    assert np.isclose(our_acc, sk_acc)


def test_accuracy_score_shape_robustness():
    """Verify accuracy_score handles (N, 1) vs (N,), PyTorch tensors, and lists without broadcasting."""
    y_true = [0, 1, 1, 0]
    y_pred_2d = [[0], [1], [1], [0]]

    # (N, 1) vs 1D list must evaluate to 1.0 (4 out of 4 correct)
    assert np.isclose(accuracy_score(y_true, y_pred_2d), 1.0)
    assert np.isclose(accuracy_score(y_pred_2d, y_true), 1.0)

    # PyTorch tensors
    t_true = torch.tensor([0, 1, 2, 3])
    t_pred = torch.tensor([0, 1, 2, 0])
    assert np.isclose(accuracy_score(t_true, t_pred), 0.75)


def test_accuracy_score_mismatched_length_raises():
    """Verify accuracy_score raises ValueError on length mismatch."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        accuracy_score([0, 1, 2], [0, 1])


def test_precision_recall_fscore_support_parity():
    """Verify precision, recall, and f1 scores match sklearn across averages."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 1, 0, 2, 2])
    y_pred = np.array([0, 2, 2, 0, 1, 1, 1, 0, 2, 0])

    for avg in ["weighted", "macro", "micro"]:
        our_p, our_r, our_f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg)
        sk_p, sk_r, sk_f, _ = sk_metrics.precision_recall_fscore_support(y_true, y_pred, average=avg)

        assert np.isclose(our_p, sk_p, atol=1e-6)
        assert np.isclose(our_r, sk_r, atol=1e-6)
        assert np.isclose(our_f, sk_f, atol=1e-6)

        assert np.isclose(precision_score(y_true, y_pred, average=avg), sk_p, atol=1e-6)
        assert np.isclose(recall_score(y_true, y_pred, average=avg), sk_r, atol=1e-6)
        assert np.isclose(f1_score(y_true, y_pred, average=avg), sk_f, atol=1e-6)


def test_precision_recall_with_unseen_prediction_class():
    """Verify metrics handle labels appearing only in predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 2])  # class 2 unseen in ground truth

    for avg in ["weighted", "macro", "micro"]:
        our_f = f1_score(y_true, y_pred, average=avg)
        sk_f = sk_metrics.f1_score(y_true, y_pred, average=avg)
        assert np.isclose(our_f, sk_f, atol=1e-6)


def test_matthews_corrcoef_parity():
    """Verify MCC matches sklearn for binary and multiclass."""
    # Binary
    y_t_bin = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    y_p_bin = np.array([0, 1, 0, 0, 1, 1, 1, 0])
    our_mcc_bin = matthews_corrcoef(y_t_bin, y_p_bin)
    sk_mcc_bin = sk_metrics.matthews_corrcoef(y_t_bin, y_p_bin)
    assert np.isclose(our_mcc_bin, sk_mcc_bin, atol=1e-6)

    # Multiclass
    y_t_multi = np.array([0, 1, 2, 0, 1, 2, 0, 2, 1])
    y_p_multi = np.array([0, 2, 2, 0, 1, 1, 0, 2, 2])
    our_mcc_multi = matthews_corrcoef(y_t_multi, y_p_multi)
    sk_mcc_multi = sk_metrics.matthews_corrcoef(y_t_multi, y_p_multi)
    assert np.isclose(our_mcc_multi, sk_mcc_multi, atol=1e-6)


def test_regression_metrics_parity():
    """Verify MSE and MAE match sklearn reference."""
    y_true = np.array([1.5, 2.7, 3.2, 4.8, 5.1])
    y_pred = np.array([1.4, 2.9, 3.0, 5.0, 4.9])

    our_mse = mean_squared_error(y_true, y_pred)
    sk_mse = sk_metrics.mean_squared_error(y_true, y_pred)
    assert np.isclose(our_mse, sk_mse)

    our_mae = mean_absolute_error(y_true, y_pred)
    sk_mae = sk_metrics.mean_absolute_error(y_true, y_pred)
    assert np.isclose(our_mae, sk_mae)

    reg_dict = compute_regression_metrics(y_pred, y_true)
    assert np.isclose(reg_dict["mse"], our_mse)
    assert np.isclose(reg_dict["mae"], our_mae)
    assert np.isclose(reg_dict["rmse"], np.sqrt(our_mse))


def test_compute_perplexity():
    """Verify perplexity computes exp(loss)."""
    assert np.isclose(compute_perplexity(0.0), 1.0)
    assert np.isclose(compute_perplexity(1.0), np.e)
    assert np.isclose(compute_perplexity(torch.tensor(2.5)), float(np.exp(2.5)))


def test_aggregate_results():
    """Verify aggregate_results computes mean, std, and lists."""
    results = [
        {"acc": 0.8, "loss": 0.4},
        {"acc": 0.9, "loss": 0.2},
    ]
    agg = aggregate_results(results)
    assert np.isclose(agg["acc_mean"], 0.85)
    assert np.isclose(agg["loss_mean"], 0.3)
    assert agg["acc_values"] == [0.8, 0.9]
