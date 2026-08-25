"""
TFLite export backend for edge deployment and mobile quantization.
Supports direct conversion via ai_edge_torch or PyTorch -> ONNX -> TensorFlow -> TFLite.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Optional

import torch

from .base import EXPORTER_REGISTRY, BaseExporter

logger = logging.getLogger(__name__)


@EXPORTER_REGISTRY.register("tflite")
class TFLiteExporter(BaseExporter):
    """
    TensorFlow Lite model exporter for edge computing devices and mobile runtimes.
    """

    def export(
        self,
        model: torch.nn.Module,
        output_path: str,
        input_example: Optional[torch.Tensor] = None,
        quantize: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Export a PyTorch model to TFLite format (.tflite).
        """
        if input_example is None:
            input_example = torch.randint(0, 1000, (1, 128))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Method 1: ai_edge_torch (official Google PyTorch -> TFLite converter)
        try:
            import ai_edge_torch  # type: ignore

            logger.info("Converting PyTorch model to TFLite using ai_edge_torch...")
            edge_model = ai_edge_torch.convert(model.eval(), (input_example,))
            edge_model.export(output_path)
            return output_path
        except ImportError:
            pass

        # Method 2: ONNX -> onnx2tf / tensorflow pipeline
        try:
            import tensorflow as tf  # type: ignore

            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_path = os.path.join(tmpdir, "model.onnx")
                torch.onnx.export(
                    model.eval(),
                    input_example,
                    onnx_path,
                    opset_version=14,
                    input_names=["input_ids"],
                    output_names=["logits"],
                )
                try:
                    import onnx2tf  # type: ignore

                    saved_model_dir = os.path.join(tmpdir, "saved_model")
                    onnx2tf.convert(
                        input_onnx_file_path=onnx_path,
                        output_folder_path=saved_model_dir,
                        non_verbose=True,
                    )
                    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
                    if quantize:
                        converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    tflite_model = converter.convert()
                    with open(output_path, "wb") as f:
                        f.write(tflite_model)
                    return output_path
                except ImportError:
                    pass
        except ImportError:
            pass

        # Clean fallback message when required compiler tools are absent
        raise ImportError(
            "TFLite export requires either 'ai-edge-torch' (recommended: pip install ai-edge-torch) "
            "or 'tensorflow' with 'onnx2tf' (pip install tensorflow onnx2tf)."
        )

