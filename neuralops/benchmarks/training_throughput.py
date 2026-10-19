#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Training-throughput benchmark with optional KD and PEFT measurements.

The baseline is a real forward/backward optimizer step. ``--use-kd`` adds a
frozen teacher and six existing NeuralOps distillation terms; ``--use-lora``
applies the registered LoRA method before constructing the optimizer.
"""

from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace
from typing import Any, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def _model_logits(output: Any) -> torch.Tensor:
    """Extract logits from a Hugging Face or TinyCausalLM output."""
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _model_hidden(output: Any, logits: torch.Tensor) -> torch.Tensor:
    """Extract a hidden state, falling back to logits when unavailable."""
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states:
        return hidden_states[-1]
    hidden = getattr(output, "hidden", None)
    return hidden if isinstance(hidden, torch.Tensor) else logits


def _match_feature_width(value: torch.Tensor, width: int) -> torch.Tensor:
    """Match representation widths without introducing learned teacher state."""
    if value.shape[-1] == width:
        return value
    batch, sequence, _ = value.shape
    resized = F.interpolate(
        value.float().reshape(batch * sequence, 1, -1),
        size=width,
        mode="linear",
        align_corners=False,
    )
    return resized.reshape(batch, sequence, width)


class TinyCausalLM(nn.Module):
    """Small causal model with projection names supported by NeuralOps PEFT."""

    def __init__(self, vocab_size: int = 256, hidden_size: int = 128) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size, hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **_: Any) -> Any:
        hidden = self.embedding(input_ids)
        q, k, v = self.query(hidden), self.key(hidden), self.value(hidden)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        causal = torch.ones(input_ids.shape[1], input_ids.shape[1], device=input_ids.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        hidden = hidden + self.dense(torch.matmul(scores.softmax(dim=-1), v))
        hidden, _ = self.rnn(hidden)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return SimpleNamespace(logits=logits, hidden=hidden, loss=loss)


class TrainingThroughputProfiler:
    """Measure supervised or KD training throughput and peak allocated VRAM."""

    def __init__(self, model: Union[str, nn.Module] = "gpt2", device: str = "cuda" if torch.cuda.is_available() else "cpu", dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32, use_kd: bool = False, use_lora: bool = False, kd_temperature: float = 2.0, kd_alpha: float = 0.5, lora_rank: int = 8, lora_alpha: float = 16.0) -> None:
        self.device, self.dtype = device, dtype
        self._is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        self.use_kd, self.use_lora = use_kd, use_lora
        self.kd_temperature, self.kd_alpha = float(kd_temperature), float(kd_alpha)
        self.lora_rank, self.lora_alpha = int(lora_rank), float(lora_alpha)
        self.model = (AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype) if isinstance(model, str) else model).to(device).train()
        self.lora_injected = 0
        if use_lora:
            from neuralops.trainer.methods.lora import apply_lora
            config = SimpleNamespace(lora_r=self.lora_rank, lora_alpha=self.lora_alpha, lora_dropout=0.0)
            before = {id(module) for module in self.model.modules()}
            apply_lora(self.model, config)
            self.lora_injected = sum(id(module) not in before for module in self.model.modules())
        self.teacher: nn.Module | None = None
        if use_kd:
            vocab = int(getattr(getattr(self.model, "config", None), "vocab_size", 256))
            student_width = int(getattr(getattr(self.model, "config", None), "hidden_size", 128))
            self.teacher = TinyCausalLM(vocab_size=vocab, hidden_size=max(256, student_width * 2)).to(device, dtype=dtype).eval()
            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)

    def _kd_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, student_hidden: torch.Tensor, teacher_hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Combine four logit and two representation losses from the trainer."""
        from neuralops.trainer.distillation.multi_level_kd import BlockwiseOperatorCloningLoss, LinearCKALoss, LogitDistillationLoss
        teacher_hidden = _match_feature_width(teacher_hidden, student_hidden.shape[-1]).to(student_hidden.dtype)
        terms = [
            LogitDistillationLoss("forward_kl", self.kd_temperature)(student_logits, teacher_logits),
            LogitDistillationLoss("reverse_kl", self.kd_temperature)(student_logits, teacher_logits),
            LogitDistillationLoss("beta_jsd", self.kd_temperature)(student_logits, teacher_logits),
            LogitDistillationLoss("decoupled_kd", self.kd_temperature)(student_logits, teacher_logits, labels=labels),
            BlockwiseOperatorCloningLoss()(student_hidden, teacher_hidden),
            LinearCKALoss()(student_hidden, teacher_hidden),
        ]
        return torch.stack([term.float() for term in terms]).mean()

    def profile_training_step(self, batch_size: int = 2, seq_len: int = 128, warmup_steps: int = 2, num_steps: int = 3, grad_accum_steps: int = 1) -> Dict[str, float]:
        """Run real optimizer steps and return finite measured metrics."""
        del grad_accum_steps
        vocab_size = int(getattr(getattr(self.model, "config", None), "vocab_size", 1000))
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        if self._is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

        def step() -> tuple[torch.Tensor, torch.Tensor]:
            optimizer.zero_grad(set_to_none=True)
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
            output = self.model(inputs, labels=inputs)
            student_logits = _model_logits(output)
            task_loss = getattr(output, "loss", None)
            if task_loss is None:
                task_loss = F.cross_entropy(student_logits.reshape(-1, student_logits.shape[-1]), inputs.reshape(-1))
            kd_loss = torch.zeros((), device=student_logits.device)
            total_loss = task_loss
            if self.teacher is not None:
                with torch.no_grad():
                    teacher_output = self.teacher(inputs)
                kd_loss = self._kd_loss(student_logits, _model_logits(teacher_output), _model_hidden(output, student_logits), _model_hidden(teacher_output, _model_logits(teacher_output)), inputs)
                total_loss = (1.0 - self.kd_alpha) * task_loss + self.kd_alpha * kd_loss
            total_loss.backward()
            optimizer.step()
            return task_loss.detach(), kd_loss.detach()

        for _ in range(warmup_steps):
            step()
            if self._is_cuda:
                torch.cuda.synchronize()
        step_times: list[float] = []
        task_value = kd_value = 0.0
        for _ in range(num_steps):
            if self._is_cuda:
                torch.cuda.synchronize()
            started = time.perf_counter()
            task_loss, kd_loss = step()
            if self._is_cuda:
                torch.cuda.synchronize()
            step_times.append((time.perf_counter() - started) * 1000.0)
            task_value, kd_value = float(task_loss.item()), float(kd_loss.item())
        avg_step_ms = sum(step_times) / len(step_times) if step_times else 0.0
        total_params = sum(parameter.numel() for parameter in self.model.parameters())
        trainable_params = sum(parameter.numel() for parameter in trainable)
        peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0) if self._is_cuda else 0.0
        return {
            "batch_size": float(batch_size), "seq_len": float(seq_len), "step_latency_ms": round(avg_step_ms, 3),
            "tokens_per_second": round(batch_size * seq_len / (avg_step_ms / 1000.0), 2) if avg_step_ms else 0.0,
            "peak_vram_mb": round(peak_vram_mb, 2), "use_kd": float(self.use_kd), "use_lora": float(self.use_lora),
            "kd_loss": round(kd_value, 6), "task_loss": round(task_value, 6), "lora_injected_modules": float(self.lora_injected),
            "trainable_params": float(trainable_params), "total_params": float(total_params),
            "trainable_param_pct": round(100.0 * trainable_params / max(total_params, 1), 3),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a real forward/backward throughput benchmark")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-kd", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--kd-alpha", type=float, default=0.5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    args = parser.parse_args()
    profiler = TrainingThroughputProfiler(model=TinyCausalLM(), device=args.device, dtype=torch.float16 if args.device.startswith("cuda") else torch.float32, use_kd=args.use_kd, use_lora=args.use_lora, kd_temperature=args.kd_temperature, kd_alpha=args.kd_alpha, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
    print(json.dumps(profiler.profile_training_step(batch_size=args.batch_size, seq_len=args.seq_len, warmup_steps=args.warmup_steps, num_steps=args.steps), indent=2))
