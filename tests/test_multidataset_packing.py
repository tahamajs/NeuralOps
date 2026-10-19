#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""


Test Suite for Multi-Dataset Streaming, Varlen Document Packing, and WSD Scheduler.

Validates:
  1. Dataset spec parsing (comma-separated datasets, subsets, and normalized weights).
  2. Multi-dataset dynamic round-robin and weighted interleaving with offline fallback.
  3. Variable-length document packing, context_len chunking, cu_seqlens, and causal shifted labels.
  4. Warmup-Stable-Decay (WSD) scheduler phases, decay ratios, and registry integration.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neuralops.data.multidataset import (
    MultiDatasetLoader,
    create_multidataset_loader,
)
from neuralops.data.packing import (
    PackedCollator,
    PackedDataset,
    pack_sequences,
)
from neuralops.trainer.schedulers import (
    WSDScheduler,
    create_wsd,
    get_lr_scheduler,
    get_scheduler,
)


def test_parse_multiple_datasets() -> None:
    """Test parsing comma-separated dataset specs, subsets, and sampling weights."""
    # 1. Comma-separated dataset names
    spec = "HuggingFaceFW/fineweb-edu,opencsg/Fineweb-Edu-Chinese-V2"
    names, subsets, weights = MultiDatasetLoader.parse_dataset_spec(spec)
    assert names == ["HuggingFaceFW/fineweb-edu", "opencsg/Fineweb-Edu-Chinese-V2"]
    assert subsets == {"HuggingFaceFW/fineweb-edu": None, "opencsg/Fineweb-Edu-Chinese-V2": None}
    assert len(weights) == 2
    assert pytest.approx(weights[0], 1e-6) == 0.5
    assert pytest.approx(weights[1], 1e-6) == 0.5

    # 2. Comma-separated subsets
    subset_spec = "sample-100BT,default"
    names, subsets, weights = MultiDatasetLoader.parse_dataset_spec(
        spec, subsets=subset_spec
    )
    assert subsets["HuggingFaceFW/fineweb-edu"] == "sample-100BT"
    assert subsets["opencsg/Fineweb-Edu-Chinese-V2"] == "default"

    # 3. Dict-based subsets
    dict_subsets = {"HuggingFaceFW/fineweb-edu": "sample-10BT"}
    names, subsets, weights = MultiDatasetLoader.parse_dataset_spec(
        spec, subsets=dict_subsets
    )
    assert subsets["HuggingFaceFW/fineweb-edu"] == "sample-10BT"
    assert subsets["opencsg/Fineweb-Edu-Chinese-V2"] is None

    # 4. Custom comma-separated and list weights with normalization
    names, subsets, weights = MultiDatasetLoader.parse_dataset_spec(
        spec, weights="0.7,0.3"
    )
    assert pytest.approx(weights[0], 1e-6) == 0.7
    assert pytest.approx(weights[1], 1e-6) == 0.3
    assert pytest.approx(sum(weights), 1e-6) == 1.0

    names, subsets, weights = MultiDatasetLoader.parse_dataset_spec(
        spec, weights=[7.0, 3.0]
    )
    assert pytest.approx(weights[0], 1e-6) == 0.7
    assert pytest.approx(weights[1], 1e-6) == 0.3

    # 5. Invalid weight counts and negative weights error checks
    with pytest.raises(ValueError):
        MultiDatasetLoader.parse_dataset_spec(spec, weights=[0.5])

    with pytest.raises(ValueError):
        MultiDatasetLoader.parse_dataset_spec("", weights=None)


def test_multidataset_interleaving() -> None:
    """Test dynamic round-robin and weighted interleaving with offline fallback."""
    # 1. Round-robin interleaving
    stream_a = [{"id": f"a_{i}"} for i in range(4)]
    stream_b = [{"id": f"b_{i}"} for i in range(4)]

    rr_results = list(
        MultiDatasetLoader.interleave_datasets(
            iterables=[stream_a, stream_b],
            weights=None,
            stopping_strategy="all_exhausted",
        )
    )
    expected_order = ["a_0", "b_0", "a_1", "b_1", "a_2", "b_2", "a_3", "b_3"]
    assert [x["id"] for x in rr_results] == expected_order

    # 2. First exhausted stopping strategy
    stream_short = [{"id": f"s_{i}"} for i in range(2)]
    stream_long = [{"id": f"l_{i}"} for i in range(6)]
    fe_results = list(
        MultiDatasetLoader.interleave_datasets(
            iterables=[stream_short, stream_long],
            stopping_strategy="first_exhausted",
        )
    )
    assert len(fe_results) == 4
    assert [x["id"] for x in fe_results] == ["s_0", "l_0", "s_1", "l_1"]

    # 3. Weighted interleaving distribution
    large_a = [{"src": "a", "val": i} for i in range(1000)]
    large_b = [{"src": "b", "val": i} for i in range(1000)]
    weighted_results = list(
        MultiDatasetLoader.interleave_datasets(
            iterables=[large_a, large_b],
            weights=[0.8, 0.2],
            seed=123,
            stopping_strategy="all_exhausted",
        )
    )
    first_200 = weighted_results[:200]
    a_count = sum(1 for x in first_200 if x["src"] == "a")
    # Expected ~160, verify within reasonable confidence interval
    assert 140 <= a_count <= 180

    # 4. Offline fallback verification
    loader = MultiDatasetLoader(
        dataset_names="SyntheticTest/A,SyntheticTest/B",
        subsets="subset1,subset2",
        weights="0.6,0.4",
        streaming=True,
        seed=42,
    )
    loader_iter = iter(loader)
    samples = [next(loader_iter) for _ in range(10)]
    assert len(samples) == 10
    assert all("text" in s for s in samples)
    ds_names = {s["dataset_name"] for s in samples}
    assert ds_names == {"SyntheticTest/A", "SyntheticTest/B"}

    # 5. create_multidataset_loader factory function
    dl = create_multidataset_loader(
        dataset_names="SyntheticTest/A,SyntheticTest/B",
        batch_size=2,
        streaming=True,
    )
    assert isinstance(dl, DataLoader)


def test_document_packing_and_cu_seqlens() -> None:
    """Test variable-length document packing, context chunking, cu_seqlens, and causal labels."""
    # Create documents of varying lengths: 5 tokens, 10 tokens, 25 tokens
    docs = [
        {"input_ids": [10, 11, 12, 13, 14]},
        {"input_ids": list(range(20, 30))},
        {"input_ids": list(range(40, 65))},
    ]

    target_seq_len = 32
    max_context_len = 16

    packed_batches = list(
        pack_sequences(
            documents=docs,
            seq_len=target_seq_len,
            context_len=max_context_len,
            pad_token_id=0,
            ignore_index=-100,
            mask_cross_document=True,
            allow_cross_boundary_split=True,
        )
    )

    # Total tokens = 5 + 10 + 25 = 40 tokens.
    # With chunking (context_len=16):
    # doc 0: len 5
    # doc 1: len 10
    # doc 2: chunk 1 (len 16), chunk 2 (len 9)
    # Total chunks = 4 (lens: 5, 10, 16, 9) -> sum = 40
    # Sequence 0: takes 5 + 10 + 16 + 1 = 32 tokens
    # Sequence 1: takes remaining 8 tokens + 24 zero padding = 32 tokens
    assert len(packed_batches) == 2

    # Verify Sequence 0
    seq0 = packed_batches[0]
    assert seq0["input_ids"].shape == (target_seq_len,)
    assert seq0["labels"].shape == (target_seq_len,)
    assert seq0["input_ids"].dtype == torch.long
    assert seq0["labels"].dtype == torch.long
    assert seq0["cu_seqlens"].dtype == torch.int32

    # cu_seqlens invariants
    cu0 = seq0["cu_seqlens"]
    assert cu0[0].item() == 0
    assert cu0[-1].item() == target_seq_len
    # strictly monotonically increasing
    diffs = cu0[1:] - cu0[:-1]
    assert torch.all(diffs > 0)

    # Document count in sequence 0
    assert seq0["num_documents"] == len(cu0) - 1

    # Verify Sequence 1 (with padding)
    seq1 = packed_batches[1]
    assert seq1["input_ids"].shape == (target_seq_len,)
    cu1 = seq1["cu_seqlens"]
    assert cu1[0].item() == 0
    assert cu1[-1].item() == target_seq_len
    # Padded positions in labels should be -100
    assert torch.all(seq1["labels"][8:] == -100)
    # Padded positions in input_ids should be pad_token_id (0)
    assert torch.all(seq1["input_ids"][8:] == 0)

    # Verify causal label shifting within a document
    # For doc 0 (len 5): input_ids are [10, 11, 12, 13, 14]
    # labels should be [11, 12, 13, 14, -100] (boundary masked)
    assert seq0["labels"][:4].tolist() == [11, 12, 13, 14]
    assert seq0["labels"][4].item() == -100

    # Test PackedDataset wrapper and PackedCollator
    packed_ds = PackedDataset(
        dataset=docs,
        seq_len=target_seq_len,
        context_len=max_context_len,
    )
    collator = PackedCollator(pad_token_id=0, ignore_index=-100)
    batch = collator(list(packed_ds))

    assert batch["input_ids"].shape == (2, target_seq_len)
    assert batch["labels"].shape == (2, target_seq_len)
    assert batch["batch_size"] == 2
    # Combined cu_seqlens starts at 0 and ends at total batch tokens (2 * 32 = 64)
    assert batch["cu_seqlens"][0].item() == 0
    assert batch["cu_seqlens"][-1].item() == 2 * target_seq_len


def test_wsd_scheduler_phases() -> None:
    """Test Warmup-Stable-Decay (WSD) scheduler phases, decay ratios, and registry factory."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    # Configure WSD: 20 warmup steps, 60 stable steps, 20 decay steps (total 100)
    scheduler = WSDScheduler(
        optimizer=optimizer,
        warmup_steps=20,
        stable_steps=60,
        decay_steps=20,
        min_lr_ratio=0.1,
        decay_type="cosine",
    )

    # 1. Warmup phase (step 0 to 19)
    # At step 0: factor = 0.0
    assert scheduler.current_phase == "warmup"
    assert pytest.approx(scheduler.get_factor(step=0), 1e-5) == 0.0
    # At step 10: factor = 0.5
    assert pytest.approx(scheduler.get_factor(step=10), 1e-5) == 0.5
    # At step 20: factor reaches peak (1.0)
    assert pytest.approx(scheduler.get_factor(step=20), 1e-5) == 1.0

    # 2. Stable phase (step 20 to 79)
    for s in [20, 40, 60, 79]:
        scheduler.last_epoch = s
        assert scheduler.current_phase == "stable"
        assert pytest.approx(scheduler.get_factor(step=s), 1e-5) == 1.0

    # 3. Decay phase (step 80 to 100)
    scheduler.last_epoch = 80
    assert scheduler.current_phase == "decay"
    assert pytest.approx(scheduler.get_factor(step=80), 1e-5) == 1.0

    # Mid decay (step 90): progress = 0.5 -> cos(pi/2) = 0 -> 0.1 + 0.5*(0.9)*1 = 0.55
    scheduler.last_epoch = 90
    assert pytest.approx(scheduler.get_factor(step=90), 1e-5) == 0.55

    # End of decay (step 100): reaches min_lr_ratio = 0.1
    scheduler.last_epoch = 100
    assert pytest.approx(scheduler.get_factor(step=100), 1e-5) == 0.1

    # Beyond total steps: clamped at min_lr_ratio
    scheduler.last_epoch = 150
    assert pytest.approx(scheduler.get_factor(step=150), 1e-5) == 0.1

    # 4. Decay ratio derivation check
    ratio_scheduler = WSDScheduler(
        optimizer=optimizer,
        warmup_steps=10,
        total_steps=100,
        decay_ratio=0.2,  # 20 decay steps, so 70 stable steps
        min_lr_ratio=0.0,
        decay_type="linear",
    )
    assert ratio_scheduler.warmup_steps == 10
    assert ratio_scheduler.decay_steps == 20
    assert ratio_scheduler.stable_steps == 70

    # Linear decay check: at progress 0.5 (step 80 + 10 = 90), factor = 0.5
    assert pytest.approx(ratio_scheduler.get_factor(step=90), 1e-5) == 0.5

    # 5. Registry and get_scheduler integration
    reg_scheduler = get_lr_scheduler(
        "wsd",
        optimizer=optimizer,
        warmup_steps=10,
        total_steps=100,
        decay_ratio=0.1,
    )
    assert isinstance(reg_scheduler, WSDScheduler)

    cli_scheduler = get_scheduler(
        name="wsd",
        optimizer=optimizer,
        warmup_steps=10,
        total_steps=100,
    )
    assert isinstance(cli_scheduler, WSDScheduler)

    decay_flag_scheduler = get_scheduler(
        decay_type="wsd",
        optimizer=optimizer,
        warmup_steps=10,
        total_steps=100,
    )
    assert isinstance(decay_flag_scheduler, WSDScheduler)
