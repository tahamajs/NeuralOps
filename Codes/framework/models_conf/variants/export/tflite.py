from .base import EXPORTER_REGISTRY, BaseExporter


@EXPORTER_REGISTRY.register("tflite")
class TFLiteExporter(BaseExporter):
    def export(self, model, output_path, **kwargs):
        # Placeholder – requires ONNX → TF → TFLite pipeline
        """Export."""
        raise NotImplementedError(
            "TFLite export requires ONNX → TF → TFLite conversion"
        )
