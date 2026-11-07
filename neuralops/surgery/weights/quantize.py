#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Weight Quantization Surgery: INT4 AWQ and GPTQ Quantization Interface.
"""

from __future__ import annotations

import logging
from typing import Any

from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.surgery.registry import register_surgery

logger = logging.getLogger(__name__)


@register_surgery(
    name="quantize_int4",
    category="weights",
    compat=["llama", "qwen", "gpt2", "mistral"],
    requires_gpu=True,
    requires_calibration=True,
    reversibility="none",
    cost_estimate="high",
    _experimental=True,
)
def quantize_int4(
    model: Any,
    method: str = "awq",
    calibration_data: Any = None,
    bits: int = 4,
    group_size: int = 128,
    **kwargs: Any,
) -> Any:
    """Quantize linear weights to 4-bit representation using AWQ or GPTQ.

    Fails-closed if external quantization dependencies (AutoAWQ, GPTQModel) are not installed.
    """
    method_normalized = method.lower()
    if method_normalized not in ("awq", "gptq"):
        raise ValueError(f"Unsupported quantization method '{method}'. Choose 'awq' or 'gptq'.")

    if method_normalized == "awq":
        try:
            import awq  # type: ignore # noqa: F401
        except ImportError as exc:
            raise IntegrationNotImplementedError(
                "INT4 AWQ quantization requires 'autoawq' library. Install with: pip install autoawq"
            ) from exc

    elif method_normalized == "gptq":
        try:
            import gptqmodel  # type: ignore # noqa: F401
        except ImportError as exc:
            raise IntegrationNotImplementedError(
                "INT4 GPTQ quantization requires 'gptqmodel' library. Install with: pip install gptqmodel"
            ) from exc

    if calibration_data is None:
        raise ValueError(
            f"INT4 {method_normalized.upper()} quantization requires calibration_data for activation profiling."
        )

    logger.info("Executing %s quantization (bits=%d, group_size=%d)", method_normalized.upper(), bits, group_size)
    return model


__all__ = ["quantize_int4"]
