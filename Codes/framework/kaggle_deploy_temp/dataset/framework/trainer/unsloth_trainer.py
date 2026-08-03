from .base_trainer import BaseTrainer
from .registry import TrainerRegistry


@TrainerRegistry.register("unsloth")
class UnslothTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Unsloth trainer not implemented yet.")

    def train(self) -> None:
        """Fit the model on the provided dataset."""
        pass

    def evaluate(self) -> None:
        """Run evaluate and return results."""
        pass
