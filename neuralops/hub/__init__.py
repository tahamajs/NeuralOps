#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Hugging Face Hub Integration Package.

Provides high-level APIs to download, reconstruct, inspect, and upload
surgeried and compressed models hosted on the Hugging Face Hub (org: tahamajs).
"""
from __future__ import annotations

from neuralops.hub.bert_zoo import (
    generate_bert_model_card_markdown,
    list_pretrained_bert_models,
    load_pretrained_bert,
)
from neuralops.hub.compressor_zoo import (
    generate_compressor_model_card_markdown,
    list_pretrained_compressed_bert_models,
    load_pretrained_compressed_bert,
)
from neuralops.hub.model_card import (
    audit_artifacts_on_disk,
    generate_unified_model_card,
)
from neuralops.hub.surgery_loader import (
    apply_decoder_attention_surgery,
    load_surgeried_model,
    normalize_base_model_id,
)
from neuralops.hub.uploader import (
    AsyncHFHubUploader,
    retry_hf_api_call,
    upload_with_retry,
)
from neuralops.models.surgical_io import save_surgical_delta

__all__ = [
    "load_surgeried_model",
    "apply_decoder_attention_surgery",
    "normalize_base_model_id",
    "save_surgical_delta",
    "generate_unified_model_card",
    "audit_artifacts_on_disk",
    "load_pretrained_bert",
    "list_pretrained_bert_models",
    "generate_bert_model_card_markdown",
    "load_pretrained_compressed_bert",
    "list_pretrained_compressed_bert_models",
    "generate_compressor_model_card_markdown",
    "AsyncHFHubUploader",
    "retry_hf_api_call",
    "upload_with_retry",
]
