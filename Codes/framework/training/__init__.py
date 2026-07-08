from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """
    Base class for all trainers in the framework.
    """

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass
