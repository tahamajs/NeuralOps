#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
LLM Inference Engine

High-throughput inference engine supporting sampling, beam search, and continuous batching.
"""

from __future__ import annotations
import inspect
import math
from typing import Generator, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "NeuralOpsInferenceEngine",
    "stream_generate",
    "speculative_generate",
    "beam_search_generate",
    "GenerationRequest",
    "ContinuousBatcher",
]


@dataclass
class GenerationRequest:
    """Represents a serving generation request with its generation state."""
    request_id: str
    input_ids: torch.Tensor
    max_new_tokens: int = 16
    generated: list[int] = field(default_factory=list)
    finished: bool = False
    is_prefill: bool = True
    past_kv: Any = None


class NeuralOpsInferenceEngine:
    """Full-featured greedy/top-k/top-p/repetition-penalty generation runtime."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Autoregressive generation with top-k, top-p, and repetition penalties."""
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, non-empty sequence]")
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if top_p is not None and not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0):
            raise ValueError("top_k must be a non-negative integer or None")
        if repetition_penalty <= 0 or not math.isfinite(repetition_penalty):
            raise ValueError("repetition_penalty must be finite and positive")
        curr = input_ids.clone()
        past_kv = None

        for _ in range(max_new_tokens):
            inp = curr if (past_kv is None or not use_cache) else curr[:, -1:]
            outputs = _call_model(
                self.model,
                inp,
                past_key_values=past_kv,
                use_cache=use_cache,
            )
            past_kv = getattr(outputs, "past_key_values", None)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            next_token_logits = logits[:, -1, :].clone() / max(temperature, 1e-5)

            if repetition_penalty != 1.0:
                for b in range(next_token_logits.shape[0]):
                    for tok in curr[b].tolist():
                        if next_token_logits[b, tok] > 0:
                            next_token_logits[b, tok] /= repetition_penalty
                        else:
                            next_token_logits[b, tok] *= repetition_penalty

            if top_k is not None and top_k > 0:
                val, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.shape[-1]))
                next_token_logits[next_token_logits < val[:, -1:]] = float("-inf")

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                for b in range(next_token_logits.shape[0]):
                    indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                    next_token_logits[b, indices_to_remove] = float("-inf")

            probs = F.softmax(next_token_logits, dim=-1)
            if temperature > 0.0:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            curr = torch.cat([curr, next_token], dim=-1)

        return curr


def stream_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 16,
) -> Generator[int, None, None]:
    """Yields generated tokens one-by-one."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("stream_generate currently supports exactly one sequence")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    curr = input_ids.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(curr)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            curr = torch.cat([curr, torch.tensor([[next_token]], device=curr.device)], dim=1)
            yield next_token


@torch.no_grad()
def speculative_generate(
    target_model: nn.Module,
    draft_model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 12,
    K: int = 3,
    gamma: int | None = None,
) -> torch.Tensor:
    """Greedy speculative decoding; sampling-distribution preservation is not provided."""
    if gamma is not None:
        K = gamma
    if not isinstance(K, int) or isinstance(K, bool) or K <= 0:
        raise ValueError("K must be a positive integer")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    curr = input_ids.clone()
    while curr.shape[1] - input_ids.shape[1] < max_new_tokens:
        draft_ids = curr.clone()
        draft_probs = []

        for _ in range(K):
            d_out = draft_model(draft_ids)
            d_logits = d_out.logits if hasattr(d_out, "logits") else d_out
            p = F.softmax(d_logits[:, -1, :], dim=-1)
            draft_probs.append(p)
            tok = torch.argmax(p, dim=-1, keepdim=True)
            draft_ids = torch.cat([draft_ids, tok], dim=-1)

        t_out = target_model(draft_ids)
        t_logits = t_out.logits if hasattr(t_out, "logits") else t_out
        target_probs = F.softmax(t_logits, dim=-1)

        n_accepted = 0
        for i in range(K):
            pos = curr.shape[1] + i - 1
            draft_tok = draft_ids[:, curr.shape[1] + i : curr.shape[1] + i + 1]
            best_target_tok = torch.argmax(target_probs[:, pos, :], dim=-1, keepdim=True)

            if torch.equal(draft_tok, best_target_tok):
                curr = torch.cat([curr, draft_tok], dim=-1)
                n_accepted += 1
                if curr.shape[1] - input_ids.shape[1] >= max_new_tokens:
                    return curr
            else:
                curr = torch.cat([curr, best_target_tok], dim=-1)
                break

        if n_accepted == K and curr.shape[1] - input_ids.shape[1] < max_new_tokens:
            extra_tok = torch.argmax(target_probs[:, -1, :], dim=-1, keepdim=True)
            curr = torch.cat([curr, extra_tok], dim=-1)

    return curr


@torch.no_grad()
def beam_search_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    beam_size: int = 4,
    max_new_tokens: int = 8,
    length_penalty: float = 1.0,
    beam_width: int | None = None,
) -> torch.Tensor:
    if beam_width is not None:
        beam_size = beam_width
    """Production beam search decoder with length normalization."""
    _B = input_ids.shape[0]
    beams = input_ids.repeat(beam_size, 1)
    scores = torch.zeros(beam_size, device=input_ids.device)
    scores[1:] = float("-inf")

    for _ in range(max_new_tokens):
        out = model(beams)
        logits = out.logits if hasattr(out, "logits") else out
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

        total_scores = log_probs + scores.unsqueeze(-1)
        vocab_size = total_scores.shape[-1]
        topk = total_scores.view(-1).topk(beam_size)

        new_scores = topk.values
        new_idx = topk.indices
        beam_idx = new_idx // vocab_size
        token_idx = new_idx % vocab_size

        beams = torch.cat([beams[beam_idx], token_idx.unsqueeze(-1)], dim=-1)
        scores = new_scores

    norm_scores = scores / (float(beams.shape[1]) ** length_penalty)
    best_beam = beams[norm_scores.argmax()].unsqueeze(0)
    return best_beam


class ContinuousBatcher:
    """Continuous dynamic batch scheduler admitting requests on the fly."""

    def __init__(self, model: nn.Module | None = None, max_batch_size: int = 8, max_seq_len: int = 2048) -> None:
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (max_batch_size, max_seq_len)
        ):
            raise ValueError("max_batch_size and max_seq_len must be positive integers")
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.pending: list[GenerationRequest] = []
        self.active: list[GenerationRequest] = []

    def add_request(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], torch.Tensor):
            req = GenerationRequest(
                request_id=args[0], input_ids=args[1], max_new_tokens=kwargs.get("max_new_tokens", 16)
            )
        elif len(args) == 1 and isinstance(args[0], tuple):
            req = GenerationRequest(
                request_id=args[0][0],
                input_ids=args[0][1],
                max_new_tokens=kwargs.get("max_new_tokens", 16),
            )
        elif len(args) == 1 and isinstance(args[0], GenerationRequest):
            req = args[0]
        elif "req_id" in kwargs:
            req = GenerationRequest(
                request_id=kwargs["req_id"],
                input_ids=kwargs["input_ids"],
                max_new_tokens=kwargs.get("max_new_tokens", 16),
            )
        else:
            raise TypeError("add_request expects (request_id, input_ids), a request tuple, or GenerationRequest")
        if not isinstance(req.request_id, str) or not req.request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(req.input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor")
        if req.input_ids.ndim != 2 or req.input_ids.shape[0] != 1 or req.input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [1, sequence_length] with a non-empty sequence")
        if req.input_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("input_ids must use an integer dtype")
        if not isinstance(req.max_new_tokens, int) or isinstance(req.max_new_tokens, bool) or req.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if req.input_ids.shape[1] + req.max_new_tokens > self.max_seq_len:
            raise ValueError("prompt plus max_new_tokens exceeds max_seq_len")
        if any(existing.request_id == req.request_id for existing in self.pending + self.active):
            raise ValueError(f"request_id {req.request_id!r} is already queued")
        self.pending.append(req)

    @torch.no_grad()
    def step(self, model: nn.Module | None = None) -> list[dict[str, Any]]:
        if model is not None:
            self.model = model
        if self.model is None:
            raise RuntimeError("ContinuousBatcher requires a model before step()")
        while len(self.active) < self.max_batch_size and self.pending:
            self.active.append(self.pending.pop(0))

        if not self.active:
            return []

        max_len = max(r.input_ids.shape[1] for r in self.active)
        padded = torch.zeros(len(self.active), max_len, dtype=torch.long, device=self.active[0].input_ids.device)
        mask = torch.zeros(len(self.active), max_len, dtype=torch.long, device=self.active[0].input_ids.device)

        for i, r in enumerate(self.active):
            l = r.input_ids.shape[1]
            padded[i, :l] = r.input_ids[0]
            mask[i, :l] = 1

        out = _call_model(self.model, padded, attention_mask=mask)

        logits = out.logits if hasattr(out, "logits") else out
        step_results = []

        for i, r in enumerate(self.active):
            last_prompt_position = r.input_ids.shape[1] - 1
            tok = int(logits[i, last_prompt_position, :].argmax().item())
            tok_tensor = torch.tensor([[tok]], device=r.input_ids.device)
            r.input_ids = torch.cat([r.input_ids, tok_tensor], dim=1)
            r.generated.append(tok)
            if len(r.generated) >= r.max_new_tokens:
                r.finished = True
            step_results.append({"id": r.request_id, "token": tok, "finished": r.finished})

        self.active = [r for r in self.active if not r.finished]
        return step_results


def _call_model(model: nn.Module, input_ids: torch.Tensor, **optional_kwargs: Any) -> Any:
    """Pass only supported optional arguments without masking model-internal errors."""
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return model(input_ids)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    accepted = {
        name: value
        for name, value in optional_kwargs.items()
        if accepts_kwargs or name in signature.parameters
    }
    return model(input_ids, **accepted)
