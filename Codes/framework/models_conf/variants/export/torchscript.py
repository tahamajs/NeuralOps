import torch
from .base import BaseExporter, EXPORTER_REGISTRY

@EXPORTER_REGISTRY.register("torchscript")
class TorchScriptExporter(BaseExporter):
    def export(self, model, output_path, input_example=None, **kwargs):
        if input_example is None:
            input_example = torch.randint(0, 1000, (1, 128))
        traced = torch.jit.trace(model, input_example)
        traced.save(output_path)
        return output_path