"""Core utilities and shared primitives for neuralops benchmarks."""
from __future__ import annotations

from neuralops.benchmarks.core.arguments import (
    add_common_args,
    add_compression_args,
    add_dataset_args,
    add_model_args,
)
from neuralops.benchmarks.core.data_loader import (
    load_glue_dataset,
    load_wikitext_dataset,
    make_dataloader,
)
from neuralops.benchmarks.core.kd import (
    KDLoss,
    build_teacher_cache,
    kd_train_step,
)
from neuralops.benchmarks.core.model_loader import (
    apply_compression,
    inject_attention_module,
    load_model_and_tokenizer,
)
from neuralops.benchmarks.core.stats import (
    bootstrap_ci,
    fdr_correction,
    paired_ttest,
)

__all__ = [
    "add_common_args",
    "add_model_args",
    "add_dataset_args",
    "add_compression_args",
    "load_model_and_tokenizer",
    "apply_compression",
    "inject_attention_module",
    "load_glue_dataset",
    "load_wikitext_dataset",
    "make_dataloader",
    "KDLoss",
    "kd_train_step",
    "build_teacher_cache",
    "paired_ttest",
    "bootstrap_ci",
    "fdr_correction",
]
