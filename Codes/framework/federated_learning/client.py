"""FL Client – local training on client data."""

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import FLConfig


class FLClientTrainer:
    """Handles local training on client data."""

    def __init__(self, config: FLConfig) -> None:
        self.config = config

    def train(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        global_model: Optional[torch.nn.Module] = None,
        previous_model: Optional[torch.nn.Module] = None,
        previous_round_model: Optional[torch.nn.Module] = None,
        control_variate: Optional[Dict[str, np.ndarray]] = None,
    ) -> torch.nn.Module:
        """
        Local training for one client.

        Args:
            model: client's local copy of the global model
            train_loader: local data loader
            global_model: current global model (for FedProx prox term, MOON contrastive loss)
            previous_model: model from previous round (for MOON)
            previous_round_model: model from the beginning of this round (for SCAFFOLD)
            control_variate: server control variate (for SCAFFOLD / FedProx)

        Returns:
            Trained model after local epochs
        """
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.server_learning_rate
        )
        model.train()

        for epoch in range(self.config.local_epochs):
            for batch in train_loader:
                inputs, labels = batch[0], batch[1]
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)

                # FedProx proximal term
                if (
                    self.config.aggregation.value == "fedprox"
                    and global_model is not None
                ):
                    prox_loss = 0.0
                    for p, gp in zip(model.parameters(), global_model.parameters()):
                        prox_loss += torch.sum((p - gp.detach()) ** 2)
                    loss += (self.config.fedprox_mu / 2) * prox_loss

                loss.backward()
                optimizer.step()

        return model

    @staticmethod
    def train_from_data(
        model: torch.nn.Module,
        data: Dict[str, torch.Tensor],
        epochs: int = 5,
        batch_size: int = 32,
    ) -> torch.nn.Module:
        """Convenience method: train directly from a dict of tensors."""
        inputs = data["x"]
        labels = data["y"]
        dataset = TensorDataset(inputs, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters())
        model.train()
        for _ in range(epochs):
            for batch in loader:
                bx, by = batch
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(bx), by)
                loss.backward()
                optimizer.step()
        return model
