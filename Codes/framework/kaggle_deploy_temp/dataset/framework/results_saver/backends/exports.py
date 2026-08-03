from typing import Any
import logging
import os

logger = logging.getLogger(__name__)

EXPORTERS = {}


def _register(fmt) -> Any:
    """ register."""
    def decorator(fn) -> Any:
        """Decorator."""
        EXPORTERS[fmt] = fn
        return fn

    return decorator


@_register("onnx")
def export_onnx(model, input_example, output_dir) -> None:
    """Export onnx."""
    import torch

    path = os.path.join(output_dir, "model.onnx")
    torch.onnx.export(model, input_example, path)
    logger.info("Exported ONNX to %s", path)


@_register("torchscript")
def export_torchscript(model, input_example, output_dir) -> None:
    """Export torchscript."""
    import torch

    traced = torch.jit.trace(model, input_example)
    path = os.path.join(output_dir, "model.pt")
    traced.save(path)
    logger.info("Exported TorchScript to %s", path)


@_register("tflite")
def export_tflite(model, input_example, output_dir) -> None:
    """Export tflite."""
    logger.warning("TFLite export requires TensorFlow – skipped")


@_register("tensorrt")
def export_tensorrt(model, input_example, output_dir) -> None:
    """Export tensorrt."""
    logger.warning("TensorRT export requires trtexec – skipped")
