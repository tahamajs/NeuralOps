# framework/experiment/manager.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from framework.experiment.config import ExperimentConfig
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, TRAINER_REGISTRY
from framework.trainer.methods.registry import METHOD_REGISTRY
from framework.trainer.losses import LOSS_REGISTRY
from framework.trainer.optimizers import OPTIMIZER_REGISTRY
from framework.trainer.schedulers import SCHEDULER_REGISTRY
from framework.callbacks import CALLBACK_REGISTRY
from framework.data.augmentation import AUGMENTATION_REGISTRY
from framework.hyperopt import HyperOptimizer
from framework.tracking import ExperimentTracker
from framework.model_registry import ModelRegistry

@dataclass
class FullExperimentConfig(ExperimentConfig):
    hpo_config: Dict[str, Any] = field(default_factory=dict)
    distributed_config: Dict[str, Any] = field(default_factory=dict)
    tracking_config: Dict[str, Any] = field(default_factory=dict)
    deployment_config: Dict[str, Any] = field(default_factory=dict)

class ExperimentManager:
    def __init__(self, config: FullExperimentConfig):
        self.config = config
        self.tracker = ExperimentTracker(config.tracking_config)

    def run(self, stop_event=None):
        # 1. Hyperparameter optimization (optional)
        if self.config.hpo_config.get("enabled"):
            best_params = self._run_hpo()
            self.config.update(best_params)

        # 2. Single training run (with optional distributed training)
        result = self._run_single(stop_event)
        # 3. Track final results
        self.tracker.finalize(result)
        # 4. Register the best model (optional)
        if self.config.deployment_config.get("register_model"):
            ModelRegistry.register(
                artifact_path=result["output_dir"],
                metrics=result["metrics"],
                tags=self.config.tags
            )
        return result

    def _run_hpo(self):
        optimizer = HyperOptimizer(self.config.hpo_config)
        return optimizer.optimize(self._run_single)

    def _run_single(self, stop_event=None):
        # Assembles everything just like the current run_training, but with all new options
        ...