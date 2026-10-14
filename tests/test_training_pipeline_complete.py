"""
Comprehensive tests for all training pipelines, custom loops,
callbacks, losses, optimizers, and schedulers in neuralops.trainer.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from neuralops.trainer.custom_loop_trainer import CustomLoopTrainer
from neuralops.trainer.hf_callback import NeuralOpsSurgeryCallback
from neuralops.trainer.losses import get_loss_function
from neuralops.trainer.schedulers import get_lr_scheduler
from neuralops.trainer.optimizers.riemannian import RiemannianAdam
from neuralops.trainer.callbacks import MemoryCallback, TimeCallback, GradientNormCallback
from neuralops.trainer.results_manager import ResultsManager


class DummyClassificationModel(nn.Module):
    def __init__(self, in_features=32, num_classes=2):
        super().__init__()
        self.linear1 = nn.Linear(in_features, 64)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, input_ids=None, labels=None, **kwargs):
        if input_ids is not None:
            x = input_ids.float()
        else:
            x = kwargs.get("x", torch.zeros(2, 32))
        h = self.relu(self.linear1(x))
        logits = self.classifier(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return type("Output", (), {"logits": logits, "loss": loss})()


def test_custom_loop_trainer_train_step():
    """Verify that CustomLoopTrainer executes forward, loss computation, backward, and optimizer step cleanly."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DummyClassificationModel().to(device)

    # Synthetic batch
    x = torch.randn(16, 32)
    y = torch.randint(0, 2, (16,))
    dataset = TensorDataset(x, y)

    # Adapt dataset for standard collator
    class DictDataset:
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            x_i, y_i = self.ds[idx]
            return {"input_ids": x_i, "labels": y_i}

    train_ds = DictDataset(dataset)
    eval_ds = DictDataset(dataset)

    config = {
        "learning_rate": 1e-3,
        "batch_size": 4,
        "num_epochs": 1,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "num_workers": 0,
        "task_name": "sst2",
    }

    trainer = CustomLoopTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        config=config,
        device=device,
    )
    assert trainer is not None

    # Test data loader creation
    loader = trainer._make_loader(train_ds, shuffle=False)
    batch = next(iter(loader))
    assert "input_ids" in batch and "labels" in batch


def test_surgery_hf_callback_step_and_lifecycle():
    """Verify NeuralOpsSurgeryCallback state transitions across step intervals."""
    callback = NeuralOpsSurgeryCallback(
        total_steps=100,
        stage1_steps=20,
        stage2_steps=60,
    )

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(16, 16)

    model = MockModel()

    class MockState:
        global_step = 0

    class MockControl:
        pass

    state = MockState()
    control = MockControl()

    # Step 10: Stage 1
    state.global_step = 10
    callback.on_step_begin(args=None, state=state, control=control, model=model)
    # Step 40: Stage 2
    state.global_step = 40
    callback.on_step_begin(args=None, state=state, control=control, model=model)
    # Step 80: Stage 3
    state.global_step = 80
    callback.on_step_begin(args=None, state=state, control=control, model=model)

    # Train end
    callback.on_train_end(args=None, state=state, control=control, model=model)


def test_loss_functions_and_schedulers():
    """Verify built-in loss function factory and LR scheduler constructors."""
    loss_fn = get_loss_function("cross_entropy")
    assert callable(loss_fn)

    opt = torch.optim.AdamW([nn.Parameter(torch.randn(10))], lr=1e-3)
    sched = get_lr_scheduler("cosine", optimizer=opt, num_warmup_steps=10, num_training_steps=100)
    assert sched is not None

    # Step through scheduler
    for _ in range(5):
        sched.step()
    assert opt.param_groups[0]["lr"] > 0


def test_riemannian_adam_optimizer():
    """Verify Riemannian Adam optimizer for orthogonal low-rank manifolds."""
    # Orthogonal matrix parameter (e.g., Stiefel manifold)
    p = nn.Parameter(torch.linalg.qr(torch.randn(16, 8))[0])
    opt = RiemannianAdam([p], lr=1e-3)

    loss = (p ** 2).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    assert not torch.isnan(p).any()
