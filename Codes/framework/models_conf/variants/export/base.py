from abc import ABC, abstractmethod

from framework.core.registry import Registry

EXPORTER_REGISTRY = Registry("exporters")


class BaseExporter(ABC):
    @abstractmethod
    def export(self, model, output_path: str, **kwargs) -> None:
        """Export the model to the specified format."""
        pass
