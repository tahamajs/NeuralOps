import torch

from .base import EXPORTER_REGISTRY, BaseExporter


@EXPORTER_REGISTRY.register("onnx")
class ONNXExporter(BaseExporter):
    def export(self, model, output_path, input_example=None, **kwargs) -> Any:
        """Export."""
        if input_example is None:
            input_example = torch.randint(0, 1000, (1, 128))
        torch.onnx.export(model, input_example, output_path, opset_version=14)
        return output_path
