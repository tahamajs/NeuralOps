"""Smoke tests for loss functions in neuralops.trainer.losses."""
import pytest
import torch


class TestLosses:
    def test_focal_loss(self):
        from neuralops.trainer import FocalLoss
        loss_fn = FocalLoss()
        logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_label_smoothing_loss(self):
        from neuralops.trainer import LabelSmoothingLoss
        loss_fn = LabelSmoothingLoss(smoothing=0.1)
        logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss)

    def test_dice_loss(self):
        from neuralops.trainer import DiceLoss
        loss_fn = DiceLoss()
        logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss)

    def test_poly_loss(self):
        from neuralops.trainer import PolyLoss
        loss_fn = PolyLoss()
        logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss)

    def test_contrastive_loss(self):
        from neuralops.trainer import ContrastiveLoss
        loss_fn = ContrastiveLoss()
        embeddings = torch.randn(2, 4, 128)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        loss = loss_fn(embeddings, labels)
        assert torch.isfinite(loss)
