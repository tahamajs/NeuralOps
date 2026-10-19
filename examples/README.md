#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
#
# NeuralOps examples

These examples are small, offline-friendly demonstrations of the public
`neuralops` API. They use a randomly initialized tiny DistilBERT configuration,
so they do not download a checkpoint or require credentials. Run them from the
repository root after installing the package:

| Script | Demonstrates | Command | Expected result |
|---|---|---|---|
| `01_attention_surgery.py` | Replacing attention with Cosformer and GLA | `python examples/01_attention_surgery.py` | Finite logits for both replacements |
| `02_model_compression.py` | SVD-LLM v2 compression and parameter accounting | `python examples/02_model_compression.py` | Compressed parameter count and reduction |
| `03_lora_finetuning.py` | Attention surgery plus LoRA and backward pass | `python examples/03_lora_finetuning.py` | Finite loss and trainable adapter count |
| `04_knowledge_distillation.py` | One teacher-student KD step | `python examples/04_knowledge_distillation.py` | Finite KL distillation loss |
| `05_fla_style_benchmarking.py` | Public benchmark API for representative modules | `python examples/05_fla_style_benchmarking.py` | Per-module status or an explicit optional-dependency note |
| `06_export_and_push.py` | Local export and safe Hub-upload handoff | `python examples/06_export_and_push.py` | Local artifact path; upload is opt-in |
| `07_custom_attention.py` | Custom attention registration and construction | `python examples/07_custom_attention.py` | Finite output from the registered module |
| `08_grpo_reasoning.py` | Preparing a reasoning model and optional TRL/GRPO path | `python examples/08_grpo_reasoning.py` | Model prepared; TRL status reported honestly |

Heavy integrations remain optional. Install `trl`, `datasets`, ONNX, or Hub
dependencies only for the corresponding workflow. The examples deliberately
avoid pretending that an unavailable optional dependency produced a benchmark.
