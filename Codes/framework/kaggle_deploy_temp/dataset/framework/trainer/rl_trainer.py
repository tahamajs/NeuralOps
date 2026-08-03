from .base_trainer import BaseTrainer
from .registry import TrainerRegistry


@TrainerRegistry.register("trl")
class RLTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("RL trainer not implemented yet.")

    def train(self) -> None:
        """Fit the model on the provided dataset."""
        pass

    def evaluate(self) -> None:
        """Run evaluate and return results."""
        pass
