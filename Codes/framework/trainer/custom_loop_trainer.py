from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
from .base_trainer import BaseTrainer
from .registry import TrainerRegistry
from .utils import train_model, evaluate_model
from .metrics import get_task_metrics

@TrainerRegistry.register("custom_loop")
class CustomLoopTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        config: Dict[str, Any],
        results_manager,
        device: torch.device,
        num_labels: int,
        is_multichoice: bool = False,
        compute_metrics_fn=None,
        **kwargs
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.results_manager = results_manager
        self.device = device
        self.num_labels = num_labels
        self.is_multichoice = is_multichoice
        self.compute_metrics_fn = compute_metrics_fn or get_task_metrics(config.get("task_name", "mrpc"))
        self.kwargs = kwargs

    def _make_loader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 16),
            shuffle=shuffle,
            collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
        )

    def train(self) -> Dict[str, Any]:
        train_loader = self._make_loader(self.train_dataset, shuffle=True)
        val_loader = self._make_loader(self.eval_dataset, shuffle=False)
        # Convert config to object with attributes
        class AttrDict:
            def __init__(self, d): self.__dict__.update(d)
        config_obj = AttrDict(self.config)
        result = train_model(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            config=config_obj,
            num_labels=self.num_labels,
            is_multichoice=self.is_multichoice,
        )
        self.results_manager.save_final_model(self.model)
        self.results_manager.save_plots()
        return result

    def evaluate(self) -> Dict[str, Any]:
        val_loader = self._make_loader(self.eval_dataset, shuffle=False)
        return evaluate_model(
            self.model,
            val_loader,
            self.device,
            self.num_labels,
            self.is_multichoice,
            compute_all_metrics=True,
        )