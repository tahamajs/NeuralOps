# framework/training/unsloth_trainer.py
from typing import Dict, Any
import logging
from .base_trainer import BaseTrainer
from .registry import TrainerRegistry
from .results_manager import ResultsManager

logger = logging.getLogger(__name__)

@TrainerRegistry.register("unsloth")
class UnslothTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        config: Dict[str, Any],
        results_manager: ResultsManager,
        **kwargs
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.results_manager = results_manager
        self.kwargs = kwargs
        self._trainer = None

    def _build(self):
        try:
            from unsloth import FastLanguageModel
            from transformers import Trainer, TrainingArguments
        except ImportError:
            raise ImportError("Unsloth not installed. Run: pip install unsloth")

        output_dir = self.results_manager.get_checkpoint_dir()
        args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.config.get("learning_rate", 2e-5),
            per_device_train_batch_size=self.config.get("batch_size", 16),
            per_device_eval_batch_size=self.config.get("batch_size", 16),
            num_train_epochs=self.config.get("epochs", 3),
            warmup_ratio=self.config.get("warmup_ratio", 0.1),
            max_grad_norm=self.config.get("max_grad_norm", 1.0),
            weight_decay=self.config.get("weight_decay", 0.01),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_dir=str(self.results_manager.log_dir),
            logging_steps=50,
            report_to="none",
        )
        self._trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )
        self._trainer = self.results_manager.wrap_trainer(self._trainer)

    def train(self):
        if self._trainer is None:
            self._build()
        return self._trainer.train()

    def evaluate(self):
        if self._trainer is None:
            self._build()
        return self._trainer.evaluate()