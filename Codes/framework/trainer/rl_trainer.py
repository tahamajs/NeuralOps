from .base_trainer import BaseTrainer
from .registry import TrainerRegistry

@TrainerRegistry.register("trl")
class RLTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("RL trainer not implemented yet.")
    def train(self):
        pass
    def evaluate(self):
        pass