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

    def test_6pillar_distillation_losses(self):
        from neuralops.trainer import (
            HiddenAngularMSELoss,
            OrthogonalityPenaltyLoss,
            LatentEntropyRegularizer,
            Unified6PillarDistillationLoss,
        )

        # Hidden Angular MSE
        s_hid = torch.randn(2, 8, 64, requires_grad=True)
        t_hid = torch.randn(2, 8, 64)
        ang_loss = HiddenAngularMSELoss()(s_hid, t_hid)
        assert torch.isfinite(ang_loss)
        assert ang_loss.item() >= 0.0
        ang_loss.backward()
        assert s_hid.grad is not None

        # Orthogonality Penalty
        W = torch.randn(64, 32, requires_grad=True)
        ortho_loss = OrthogonalityPenaltyLoss()(W)
        assert torch.isfinite(ortho_loss)
        assert ortho_loss.item() >= 0.0
        ortho_loss.backward()
        assert W.grad is not None

        # Latent Entropy Regularizer
        hid = torch.randn(2, 8, 64, requires_grad=True)
        ent_loss = LatentEntropyRegularizer()(hid)
        assert torch.isfinite(ent_loss)
        ent_loss.backward()
        assert hid.grad is not None

        # Unified 6-Pillar Distillation Loss
        s_logits = torch.randn(2, 8, 100, requires_grad=True)
        t_logits = torch.randn(2, 8, 100)
        s_val = torch.randn(2, 4, 8, 16)
        t_val = torch.randn(2, 4, 8, 16)
        unified_fn = Unified6PillarDistillationLoss()
        out = unified_fn(
            student_logits=s_logits,
            teacher_logits=t_logits,
            student_hidden=s_hid,
            teacher_hidden=t_hid,
            student_values=s_val,
            teacher_values=t_val,
            projection_weights=[W],
        )
        assert "loss_total" in out
        assert torch.isfinite(out["loss_total"])
        out["loss_total"].backward()

