#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Serving Request Batcher & Continuous Batch Scheduler

Provides dynamic sequence slotting, continuous batching queues, left-padded
batch assembly, and streaming iterators for high-throughput serving inference.
"""

from __future__ import annotations

import collections
import inspect
import json
import logging
import math
import queue
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.Serving.RequestBatcher")


class TGIRequest:
    """Represents a single in-flight generation request in the continuous batch."""

    def __init__(
        self,
        request_id: str,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        stop_sequences: Optional[List[str]] = None,
        temperature: float = 0.0,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1 or input_ids.numel() == 0:
            raise ValueError("input_ids must be a non-empty 1D tensor")
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if stop_sequences is not None and any(not isinstance(stop, str) for stop in stop_sequences):
            raise TypeError("stop_sequences must contain strings")
        self.request_id = request_id
        self.input_ids = input_ids
        self.max_new_tokens = max_new_tokens
        self.stop_sequences = stop_sequences or []
        self.temperature = temperature
        self.top_k = top_k
        self.eos_token_id = eos_token_id

        self.generated_ids: List[int] = []
        self.past_kv: Optional[Any] = None
        self.slot_id: Optional[int] = None
        self.finished: bool = False
        self.finish_reason: Optional[str] = None
        self.created_at: float = time.time()
        self.stream_queue: queue.Queue[Optional[Tuple[int, str, bool]]] = queue.Queue()


TGIRequestSlot = TGIRequest


class TGIRequestBatch:
    """Manages active and queued requests with dynamic sequence slotting and batch assembly."""

    def __init__(self, max_batch_size: int = 32, max_seq_len: int = 2048):
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (max_batch_size, max_seq_len)
        ):
            raise ValueError("max_batch_size and max_seq_len must be positive integers")
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.active_slots: Dict[str, TGIRequest] = {}
        self.queued_slots: collections.deque[TGIRequest] = collections.deque()
        self.completed_slots: Dict[str, TGIRequest] = {}
        self.free_slot_ids: List[int] = list(range(max_batch_size))
        self.slot_to_request_id: Dict[int, str] = {}

    def add(self, request: TGIRequest) -> None:
        if request.request_id in self.active_slots or request.request_id in self.completed_slots:
            raise ValueError(f"request_id {request.request_id!r} has already been used")
        if any(queued.request_id == request.request_id for queued in self.queued_slots):
            raise ValueError(f"request_id {request.request_id!r} is already queued")
        if request.input_ids.numel() + request.max_new_tokens > self.max_seq_len:
            raise ValueError("prompt plus max_new_tokens exceeds max_seq_len")
        if len(self.active_slots) < self.max_batch_size and self.free_slot_ids:
            slot_id = self.free_slot_ids.pop(0)
            request.slot_id = slot_id
            self.slot_to_request_id[slot_id] = request.request_id
            self.active_slots[request.request_id] = request
        else:
            self.queued_slots.append(request)

    def remove(self, request_id: str) -> None:
        if request_id in self.active_slots:
            slot = self.active_slots.pop(request_id)
            slot.past_kv = None
            if slot.slot_id is not None:
                self.slot_to_request_id.pop(slot.slot_id, None)
                self.free_slot_ids.append(slot.slot_id)
                self.free_slot_ids.sort()
                slot.slot_id = None
            self.completed_slots[request_id] = slot
        self.admit_queued()

    def active_request_ids(self) -> List[str]:
        return [r_id for r_id, s in self.active_slots.items() if not s.finished]

    def admit_queued(self) -> None:
        while self.queued_slots and len(self.active_slots) < self.max_batch_size and self.free_slot_ids:
            slot = self.queued_slots.popleft()
            slot_id = self.free_slot_ids.pop(0)
            slot.slot_id = slot_id
            self.slot_to_request_id[slot_id] = slot.request_id
            self.active_slots[slot.request_id] = slot

    def get_slot_id(self, request_id: str) -> Optional[int]:
        """Returns the assigned batch slot index (0..max_batch_size-1) for an active request."""
        slot = self.active_slots.get(request_id)
        return slot.slot_id if slot is not None else None

    def get_request_by_slot(self, slot_id: int) -> Optional[TGIRequest]:
        """Returns the active request occupying the given slot index."""
        req_id = self.slot_to_request_id.get(slot_id)
        return self.active_slots.get(req_id) if req_id is not None else None

    def slot_table(self) -> Dict[int, Optional[str]]:
        """Returns mapping of all slot indices (0..max_batch_size-1) to request IDs."""
        return {slot_idx: self.slot_to_request_id.get(slot_idx) for slot_idx in range(self.max_batch_size)}

    def assemble_batch(
        self,
        pad_token_id: int = 0,
        padding_side: str = "left",
    ) -> Dict[str, Any]:
        """
        Assembles active requests into a padded batch tensor with attention masks and slot indices.

        Args:
            pad_token_id: Token ID used for padding sequences of unequal length.
            padding_side: "left" (standard for causal autoregressive generation) or "right".

        Returns:
            Dictionary containing:
            - input_ids: Tensor [B, max_len] of padded prompt/context token IDs.
            - attention_mask: Tensor [B, max_len] of 1 (real token) and 0 (padded token).
            - request_ids: List[str] of request IDs corresponding to each row.
            - slot_ids: List[int] of slot indices assigned to each row.
            - seq_lens: List[int] of unpadded sequence lengths.
            - max_len: Maximum unpadded sequence length across active requests.
        """
        active = [s for s in self.active_slots.values() if not s.finished and s.input_ids is not None]
        if not active:
            return {
                "input_ids": torch.empty((0, 0), dtype=torch.long),
                "attention_mask": torch.empty((0, 0), dtype=torch.long),
                "request_ids": [],
                "slot_ids": [],
                "seq_lens": [],
                "max_len": 0,
            }

        seq_lens = [len(s.input_ids) for s in active]
        max_len = max(seq_lens)
        device = active[0].input_ids.device
        batch_size = len(active)

        padded_inputs = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        request_ids = []
        slot_ids = []

        for i, s in enumerate(active):
            slen = seq_lens[i]
            request_ids.append(s.request_id)
            slot_ids.append(s.slot_id if s.slot_id is not None else i)
            if padding_side == "left":
                padded_inputs[i, max_len - slen :] = s.input_ids
                attention_mask[i, max_len - slen :] = 1
            else:
                padded_inputs[i, :slen] = s.input_ids
                attention_mask[i, :slen] = 1

        return {
            "input_ids": padded_inputs,
            "attention_mask": attention_mask,
            "request_ids": request_ids,
            "slot_ids": slot_ids,
            "seq_lens": seq_lens,
            "max_len": max_len,
        }

    def assemble_decode_batch(
        self,
        active_ids: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, List[str], List[int]]:
        """
        Assembles single-step decode tokens into a unified [B, 1] tensor across active slots.

        Args:
            active_ids: Optional subset of request IDs to assemble. If None, uses all active requests.

        Returns:
            Tuple of:
            - batched_tokens: Tensor [B, 1] with the most recent token for each request.
            - request_ids: List[str] of request IDs in row order.
            - slot_ids: List[int] of slot indices corresponding to each row.
        """
        if active_ids is None:
            active_ids = self.active_request_ids()

        tokens: List[int] = []
        req_ids: List[str] = []
        slot_ids: List[int] = []
        device = None

        for r_id in active_ids:
            slot = self.active_slots.get(r_id)
            if not slot or slot.finished or slot.input_ids is None:
                continue
            if device is None:
                device = slot.input_ids.device
            last_tok = slot.generated_ids[-1] if slot.generated_ids else int(slot.input_ids[-1].item())
            tokens.append(last_tok)
            req_ids.append(r_id)
            slot_ids.append(slot.slot_id if slot.slot_id is not None else len(slot_ids))

        if not tokens:
            return torch.empty((0, 1), dtype=torch.long), [], []

        tensor_device = device if device is not None else torch.device("cpu")
        batched_tokens = torch.tensor(tokens, dtype=torch.long, device=tensor_device).unsqueeze(1)
        return batched_tokens, req_ids, slot_ids

    def get_batched_inputs(self, pad_token_id: int = 0) -> torch.Tensor:
        """Returns dynamically left-padded batch tensor [B, max_len] for active requests."""
        batch_dict = self.assemble_batch(pad_token_id=pad_token_id, padding_side="left")
        return batch_dict["input_ids"]


class TGIStreamingIterator:
    """Iterator yielding (request_id, token, text, is_final) tuples per request."""

    def __init__(self, request: TGIRequest):
        self.request = request

    def __iter__(self) -> Iterator[Tuple[str, int, str, bool]]:
        return self

    def __next__(self) -> Tuple[str, int, str, bool]:
        item = self.request.stream_queue.get()
        if item is None:
            raise StopIteration
        token, text, is_final = item
        return (self.request.request_id, token, text, is_final)


class TGIServingAdapter:
    """Lightweight model-backed continuous generation adapter with dynamic sequence slotting.

    Provides continuous dynamic batch scheduling, per-sequence slot assignment,
    dynamic left-padded prompt assembly, and token streaming iterators.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any = None,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        device: Union[str, torch.device] = "cuda",
        eos_token_id: Optional[int] = None,
    ):
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (max_batch_size, max_seq_len)
        ):
            raise ValueError("max_batch_size and max_seq_len must be positive integers")
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id if eos_token_id is not None else getattr(tokenizer, "eos_token_id", None)

        self.batch_manager = TGIRequestBatch(max_batch_size=max_batch_size, max_seq_len=max_seq_len)

        self._total_requests = 0
        self._total_tokens_generated = 0
        self._step_times: collections.deque[float] = collections.deque(maxlen=100)

    def stats(self) -> Dict[str, Any]:
        """Operational statistics for monitoring and health endpoints."""
        avg_step_ms = (sum(self._step_times) / len(self._step_times) * 1000.0) if self._step_times else 0.0
        tokens_per_sec = (1000.0 / avg_step_ms) if avg_step_ms > 0 else 0.0
        return {
            "total_requests": self._total_requests,
            "active_requests": len(self.batch_manager.active_request_ids()),
            "active_slots": len(self.batch_manager.active_slots),
            "free_slots": len(self.batch_manager.free_slot_ids),
            "max_batch_size": self.max_batch_size,
            "queued_requests": len(self.batch_manager.queued_slots),
            "completed_requests": len(self.batch_manager.completed_slots),
            "total_tokens_generated": self._total_tokens_generated,
            "avg_latency_ms": round(avg_step_ms, 2),
            "avg_tokens_per_sec": round(tokens_per_sec, 2),
            "device": str(self.device),
        }

    def clear_finished(self) -> None:
        """Explicitly clear finished requests and free per-request KV tensors."""
        finished_ids = [s_id for s_id, s in self.batch_manager.active_slots.items() if s.finished]
        for s_id in finished_ids:
            self.batch_manager.remove(s_id)

    def _tokenize_prompt(self, prompt: Union[str, List[int], torch.Tensor]) -> torch.Tensor:
        """Convert input prompt into an integer 1D tensor [seq_len]."""
        if isinstance(prompt, torch.Tensor):
            t = prompt.squeeze()
            if t.dim() == 0:
                t = t.unsqueeze(0)
            return t.to(self.device, dtype=torch.long)
        if isinstance(prompt, list):
            return torch.tensor(prompt, dtype=torch.long, device=self.device)
        if isinstance(prompt, str):
            if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
                ids = self.tokenizer.encode(prompt)
                return torch.tensor(ids, dtype=torch.long, device=self.device)
            ids = [ord(c) % 50257 for c in prompt] or [0]
            return torch.tensor(ids, dtype=torch.long, device=self.device)
        raise TypeError(f"Unsupported prompt format: {type(prompt)}")

    def _decode_token(self, token_id: int) -> str:
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode([token_id], skip_special_tokens=False)
        return chr(token_id % 128) if 32 <= (token_id % 128) <= 126 else f"<{token_id}>"

    def add_request(
        self,
        request_id: str,
        prompt: Union[str, List[int], torch.Tensor],
        max_new_tokens: int = 16,
        stop_sequences: Optional[List[str]] = None,
        temperature: float = 0.0,
        top_k: int = 50,
        **kwargs: Any,
    ) -> TGIRequest:
        """Enqueues a new request for continuous generation."""
        token_tensor = self._tokenize_prompt(prompt)
        req = TGIRequest(
            request_id=request_id,
            input_ids=token_tensor,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=self.eos_token_id,
        )
        self.batch_manager.add(req)
        self._total_requests += 1
        return req

    def prefill(self, request_ids: List[str]) -> Dict[str, Any]:
        """
        Executes prefill phase for newly scheduled requests.
        Processes prompt tokens and returns initial KV caches.
        """
        results: Dict[str, Any] = {}
        for req_id in request_ids:
            slot = self.batch_manager.active_slots.get(req_id)
            if not slot or slot.input_ids is None or slot.past_kv is not None or slot.finished:
                continue

            inp = slot.input_ids.unsqueeze(0).to(self.device)
            outputs = _call_model(self.model, inp, use_cache=True)
            logits = outputs.logits[:, -1, :] if hasattr(outputs, "logits") else outputs[:, -1, :]
            pkv = getattr(outputs, "past_key_values", None)

            slot.past_kv = pkv
            next_tok = self._sample_token(logits, slot.temperature, slot.top_k)
            slot.generated_ids.append(next_tok)
            self._total_tokens_generated += 1
            text_piece = self._decode_token(next_tok)

            is_max = len(slot.generated_ids) >= slot.max_new_tokens
            is_stop = self.check_stop(slot, next_tok)
            if is_max or is_stop:
                slot.finished = True
                slot.finish_reason = "max_tokens" if is_max else "stop"
                slot.stream_queue.put((next_tok, text_piece, True))
                slot.stream_queue.put(None)
            else:
                slot.stream_queue.put((next_tok, text_piece, False))

            results[req_id] = pkv

        return results

    def decode_step(self, active_ids: List[str]) -> Dict[str, int]:
        """
        Executes a single continuous decode step across all active requests.
        Outputs 1 new token per active request and updates its KV cache.
        """
        step_t0 = time.time()
        new_tokens: Dict[str, int] = {}

        needs_prefill = [
            r_id for r_id in active_ids
            if r_id in self.batch_manager.active_slots
            and self.batch_manager.active_slots[r_id].past_kv is None
            and not self.batch_manager.active_slots[r_id].finished
        ]
        if needs_prefill:
            self.prefill(needs_prefill)

        prefetched_ids = set(needs_prefill)
        for req_id in needs_prefill:
            slot = self.batch_manager.active_slots.get(req_id)
            if slot is not None and slot.generated_ids:
                new_tokens[req_id] = slot.generated_ids[-1]

        for req_id in active_ids:
            if req_id in prefetched_ids:
                continue
            slot = self.batch_manager.active_slots.get(req_id)
            if not slot or slot.finished:
                continue

            last_tok = slot.generated_ids[-1] if slot.generated_ids else int(slot.input_ids[-1].item())
            inp = torch.tensor([[last_tok]], dtype=torch.long, device=self.device)

            if slot.past_kv is not None:
                outputs = _call_model(self.model, inp, past_key_values=slot.past_kv, use_cache=True)
                logits = outputs.logits[:, -1, :] if hasattr(outputs, "logits") else outputs[:, -1, :]
                slot.past_kv = getattr(outputs, "past_key_values", None)
            else:
                full_context = torch.cat([
                    slot.input_ids.to(self.device),
                    torch.tensor(slot.generated_ids, dtype=torch.long, device=self.device),
                ])
                outputs = _call_model(self.model, full_context.unsqueeze(0))
                logits = outputs.logits[:, -1, :] if hasattr(outputs, "logits") else outputs[:, -1, :]

            next_tok = self._sample_token(logits, slot.temperature, slot.top_k)
            slot.generated_ids.append(next_tok)
            new_tokens[req_id] = next_tok
            self._total_tokens_generated += 1
            text_piece = self._decode_token(next_tok)

            is_max = len(slot.generated_ids) >= slot.max_new_tokens
            is_stop = self.check_stop(slot, next_tok)

            if is_max or is_stop:
                slot.finished = True
                slot.finish_reason = "max_tokens" if is_max else "stop_sequence"
                slot.stream_queue.put((next_tok, text_piece, True))
                slot.stream_queue.put(None)
            else:
                slot.stream_queue.put((next_tok, text_piece, False))

        self.clear_finished()
        self._step_times.append(time.time() - step_t0)
        return new_tokens

    def check_stop(self, request: TGIRequest, new_token: int) -> bool:
        """Determines whether generation should cease due to EOS or stop sequences."""
        if request.eos_token_id is not None and new_token == request.eos_token_id:
            return True

        if request.stop_sequences:
            if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                text_so_far = self.tokenizer.decode(request.generated_ids, skip_special_tokens=False)
            else:
                text_so_far = "".join(self._decode_token(t) for t in request.generated_ids)
            for stop in request.stop_sequences:
                if stop in text_so_far:
                    return True
        return False

    def _sample_token(self, logits: torch.Tensor, temperature: float = 0.0, top_k: int = 50) -> int:
        """Greedy argmax or top-k filtered categorical sampling."""
        if temperature <= 1e-4:
            return int(torch.argmax(logits, dim=-1).item())

        scaled = logits / temperature
        if top_k > 0:
            v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
            scaled[scaled < v[:, [-1]]] = float("-inf")
        probs = torch.softmax(scaled, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return int(sampled.item())

    def continuous_batching_generate(
        self,
        prompts: List[Union[str, List[int], torch.Tensor]],
        max_new_tokens: int = 16,
        **kwargs: Any,
    ) -> List[str]:
        """
        High-level blocking API: Submits multiple prompts, executes continuous
        batching until all sequences terminate, and returns list of completions.
        """
        slots = []
        for i, p in enumerate(prompts):
            req_id = f"req_{int(time.time() * 1000)}_{self._total_requests}_{i}"
            slot = self.add_request(req_id, p, max_new_tokens=max_new_tokens, **kwargs)
            slots.append(slot)

        while any(not s.finished for s in slots):
            active_ids = self.batch_manager.active_request_ids()
            if not active_ids:
                break
            self.decode_step(active_ids)

        completions = []
        for s in slots:
            if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                text = self.tokenizer.decode(s.generated_ids, skip_special_tokens=True)
            else:
                text = "".join(self._decode_token(t) for t in s.generated_ids)
            completions.append(text)
        return completions

    def streaming_generate(
        self,
        prompt: Union[str, List[int], torch.Tensor],
        max_new_tokens: int = 16,
        **kwargs: Any,
    ) -> Iterator[Tuple[int, str]]:
        """
        Streaming API: Generates tokens incrementally and yields (token_id, token_str) tuples.
        """
        req_id = f"stream_{int(time.time() * 1000)}"
        slot = self.add_request(req_id, prompt, max_new_tokens=max_new_tokens, **kwargs)

        while not slot.finished or not slot.stream_queue.empty():
            if not slot.finished:
                self.decode_step([slot.request_id])
            try:
                item = slot.stream_queue.get_nowait()
                if item is None:
                    break
                tok, text, _ = item
                yield (tok, text)
            except queue.Empty:
                if slot.finished:
                    break

    def standalone_server_hook(self) -> Any:
        """
        Returns a minimal FastAPI adapter exposing TGI-style endpoints:
        - GET  /health
        - GET  /info
        - POST /generate
        - POST /generate_stream
        """
        try:
            from fastapi import FastAPI, Request
            from fastapi.responses import StreamingResponse
        except ImportError:
            raise ImportError(
                "FastAPI is required to run the standalone TGI server hook. "
                "Install with: pip install fastapi uvicorn"
            )

        app = FastAPI(title="NeuralOps TGI Serving Adapter", version="1.4.2")

        @app.get("/health")
        def health() -> Dict[str, str]:
            return {"status": "ok"}

        @app.get("/info")
        def info() -> Dict[str, Any]:
            return self.stats()

        @app.post("/generate")
        async def generate(request: Request) -> Dict[str, Any]:
            data = await request.json()
            if not isinstance(data, dict):
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail="request body must be a JSON object")
            prompt = data.get("inputs", "")
            params = data.get("parameters", {})
            if not isinstance(params, dict):
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail="parameters must be a JSON object")
            max_new = params.get("max_new_tokens", 16)
            completions = self.continuous_batching_generate([prompt], max_new_tokens=max_new)
            return {"generated_text": completions[0]}

        @app.post("/generate_stream")
        async def generate_stream(request: Request) -> StreamingResponse:
            data = await request.json()
            if not isinstance(data, dict):
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail="request body must be a JSON object")
            prompt = data.get("inputs", "")
            params = data.get("parameters", {})
            if not isinstance(params, dict):
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail="parameters must be a JSON object")
            max_new = params.get("max_new_tokens", 16)

            def event_generator():
                for tok, text in self.streaming_generate(prompt, max_new_tokens=max_new):
                    yield f"data: {json.dumps({'token': tok, 'text': text})}\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        return app


def _call_model(model: nn.Module, input_ids: torch.Tensor, **optional_kwargs: Any) -> Any:
    """Pass only supported model kwargs without catching errors raised inside forward."""
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return model(input_ids)
    accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    accepted = {
        name: value
        for name, value in optional_kwargs.items()
        if accepts_kwargs or name in signature.parameters
    }
    return model(input_ids, **accepted)


__all__ = [
    "TGIRequest",
    "TGIRequestSlot",
    "TGIRequestBatch",
    "TGIStreamingIterator",
    "TGIServingAdapter",
]
