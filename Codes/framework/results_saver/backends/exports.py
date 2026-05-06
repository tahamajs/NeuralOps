import logging
logger = logging.getLogger(__name__)

EXPORTERS = {}

def _register(fmt):
    def decorator(fn):
        EXPORTERS[fmt] = fn
        return fn
    return decorator

@_register("onnx")
def export_onnx(model, input_example, output_dir):
    import torch
    path = f"{output_dir}/model.onnx"
    torch.onnx.export(model, input_example, path)
    logger.info("Exported ONNX to %s", path)

@_register("torchscript")
def export_torchscript(model, input_example, output_dir):
    import torch
    traced = torch.jit.trace(model, input_example)
    path = f"{output_dir}/model.pt"
    traced.save(path)
    logger.info("Exported TorchScript to %s", path)

@_register("tflite")
def export_tflite(model, input_example, output_dir):
    logger.warning("TFLite export requires TensorFlow – skipped")

@_register("tensorrt")
def export_tensorrt(model, input_example, output_dir):
    logger.warning("TensorRT export requires trtexec – skipped")