import torch
from .base import BaseExporter, EXPORTER_REGISTRY

@EXPORTER_REGISTRY.register("tensorrt")
class TensorRTExporter(BaseExporter):
    def export(self, model, output_path, input_example=None, **kwargs):
        try:
            import torch_tensorrt
        except ImportError:
            raise ImportError("Install torch-tensorrt for TensorRT export")
        if input_example is None:
            input_example = torch.randint(0, 1000, (1, 128))
        trt_model = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input(shape=input_example.shape)])
        torch.jit.save(trt_model, output_path)
        return output_path