from .base_trainer import BaseTrainer
from .registry import TrainerRegistry

@TrainerRegistry.register("unsloth")
class UnslothTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Unsloth trainer not implemented yet.")
    def train(self):
        pass
    def evaluate(self):
        pass