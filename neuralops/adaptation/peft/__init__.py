#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parameter-Efficient Fine-Tuning (PEFT) and Adapter Recovery Facade.

Canonical Owners:
- LoRA: neuralops.trainer.methods.lora
- Operation Contract: neuralops.operations.adaptation.lora
- SVD Adapters: neuralops.trainer.peft_recovery.svd_adapters
"""

from __future__ import annotations

from neuralops.operations.adaptation.lora import CONTRACT as LORA_OPERATION_CONTRACT
from neuralops.trainer.methods.lora import LoRALinear
from neuralops.trainer.peft_recovery.svd_adapters import DoRALinear, PiSSALinear, inject_pissa_adapters

__all__ = [
    "LoRALinear",
    "PiSSALinear",
    "DoRALinear",
    "inject_pissa_adapters",
    "LORA_OPERATION_CONTRACT",
]
