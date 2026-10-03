"""Unit Tests for Extended Dataset Loaders (CoLA, STS-B, LRA ListOps, CIFAR-10)."""

import pytest
import torch
from framework.datasets_local.extended_lra_loaders import (
    ListOpsDataset, SequentialCIFAR10Dataset, get_extended_task_dataset)


def test_listops_dataset_shapes():
    ds = ListOpsDataset(num_samples=10, seq_len=128, num_classes=10)
    assert len(ds) == 10
    sample = ds[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert sample["input_ids"].shape == (128,)
    assert 0 <= sample["labels"].item() < 10


def test_cifar10_dataset_shapes():
    ds = SequentialCIFAR10Dataset(num_samples=10, seq_len=128, num_classes=10)
    assert len(ds) == 10
    sample = ds[0]
    assert sample["input_ids"].shape == (128,)
    assert 0 <= sample["input_ids"].max().item() <= 255


def test_get_extended_task_dataset_glue():
    ds_cola = get_extended_task_dataset("cola", num_samples=8, seq_len=64)
    assert len(ds_cola) == 8
    sample_cola = ds_cola[0]
    assert "input_ids" in sample_cola
    assert "labels" in sample_cola

    ds_stsb = get_extended_task_dataset("stsb", num_samples=8, seq_len=64)
    assert len(ds_stsb) == 8
    sample_stsb = ds_stsb[0]
    assert "input_ids" in sample_stsb
