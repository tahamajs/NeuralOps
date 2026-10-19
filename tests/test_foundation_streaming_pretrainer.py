#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neuralops.data.streaming import PackedTokenStreamDataset
from neuralops.trainer.foundation_pretrainer import InfiniteContextFoundationPretrainer


def test_packed_token_stream_dataset_packing_and_checkpoint():
    """Verify document packing into fixed chunks and exact state checkpoint resumption."""
    docs = [
        list(range(1, 11)),
        list(range(11, 26)),
        list(range(26, 51)),
        list(range(51, 81)),
        list(range(81, 101)),
    ]

    def doc_iter():
        for d in docs:
            yield d

    dataset = PackedTokenStreamDataset(
        token_iterator_fn=doc_iter,
        seq_len=16,
        eos_token_id=999,
        dp_rank=0,
        dp_world_size=1,
    )

    chunks = list(dataset)
    assert len(chunks) > 0
    for c in chunks:
        assert c["input_ids"].shape == (16,)
        assert c["labels"].shape == (16,)
        assert c["is_eos"].shape == (16,)

    state = dataset.state_dict()
    assert state["chunk_counter"] == len(chunks)

    resumed_dataset = PackedTokenStreamDataset(
        token_iterator_fn=doc_iter,
        seq_len=16,
        eos_token_id=999,
    )
    resumed_dataset.load_state_dict(state)
    resumed_chunks = list(resumed_dataset)
    assert len(resumed_chunks) == 0


def test_infinite_context_foundation_pretrainer_stateful_step():
    """Verify state persistence across chunks and curriculum alpha annealing."""
    class DummyRecurrentLM(nn.Module):
        def __init__(self, vocab_size=100, d_model=32):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.rnn = nn.GRU(d_model, d_model, batch_first=True)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, input_ids):
            x = self.embed(input_ids)
            out, h = self.rnn(x)
            logits = self.head(out)
            return logits, None, h

    model = DummyRecurrentLM()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pretrainer = InfiniteContextFoundationPretrainer(
        model=model,
        optimizer=opt,
        warmup_steps=10,
        max_steps=50,
        device="cpu",
    )

    assert pretrainer.get_curriculum_alpha(0) == 0.0
    assert pretrainer.get_curriculum_alpha(5) == 0.5
    assert pretrainer.get_curriculum_alpha(10) == 1.0

    for i in range(3):
        batch = {
            "input_ids": torch.randint(0, 100, (1, 16)),
            "labels": torch.randint(0, 100, (1, 16)),
            "is_eos": torch.zeros((1, 16), dtype=torch.bool),
        }
        loss_val = pretrainer.train_step(batch)
        assert isinstance(loss_val, float)
        assert loss_val > 0.0
        assert pretrainer.persistent_state is not None
        assert pretrainer.persistent_state.shape == (1, 1, 32)
