"""
Text Generation Inference Integration

Provides serving hooks and model architecture adapters for Text Generation Inference (TGI).

Author: Taha Majlesi
"""

from __future__ import annotations

import collections
import logging
import queue
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.TGI")


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
        self.request_id = request_id
        # input_ids: [L] 1D long tensor
        self.input_ids = input_ids
        self.max_new_tokens = max_new_tokens
        self.stop_sequences = stop_sequences or []
        self.temperature = temperature
        self.top_k = top_k
        self.eos_token_id = eos_token_id

        self.generated_ids: List[int] = []
        self.past_kv: Optional[Any] = None
        self.finished: bool = False
        self.finish_reason: Optional[str] = None
        self.created_at: float = time.time()
        self.stream_queue: queue.Queue[Optional[Tuple[int, str, bool]]] = queue.Queue()
 
 
# Alias for backwards compatibility
TGIRequestSlot = TGIRequest


class TGIRequestBatch:
    """Manages a dynamic pool of active and queued requests under continuous batching."""

    def __init__(self, max_batch_size: int = 32, max_seq_len: int = 2048):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.active_slots: Dict[str, TGIRequest] = {}
        self.queued_slots: collections.deque[TGIRequest] = collections.deque()
        self.completed_slots: Dict[str, TGIRequest] = {}

    def add(self, request: TGIRequest) -> None:
        if len(self.active_slots) < self.max_batch_size:
            self.active_slots[request.request_id] = request
        else:
            self.queued_slots.append(request)

    def remove(self, request_id: str) -> None:
        if request_id in self.active_slots:
            slot = self.active_slots.pop(request_id)
            slot.past_kv = None
            self.completed_slots[request_id] = slot
        self.admit_queued()

    def active_request_ids(self) -> List[str]:
        return [r_id for r_id, s in self.active_slots.items() if not s.finished]

    def admit_queued(self) -> None:
        while self.queued_slots and len(self.active_slots) < self.max_batch_size:
            slot = self.queued_slots.popleft()
            self.active_slots[slot.request_id] = slot

    def get_batched_inputs(self, pad_token_id: int = 0) -> torch.Tensor:
        """Returns dynamically left-padded batch tensor [B, max_len] for active requests."""
        active = [s for s in self.active_slots.values() if not s.finished and s.input_ids is not None]
        if not active:
            return torch.empty((0, 0), dtype=torch.long)
        max_len = max(len(s.input_ids) for s in active)
        batch = []
        for s in active:
            pad_len = max_len - len(s.input_ids)
            if pad_len > 0:
                pad = torch.full((pad_len,), pad_token_id, dtype=torch.long, device=s.input_ids.device)
                batch.append(torch.cat([pad, s.input_ids]))
            else:
                batch.append(s.input_ids)
        return torch.stack(batch, dim=0)


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
    """
    Continuous batching and streaming generation engine compatible with TGI servers.

    Features:
    - True iteration-level continuous batching: new requests are admitted every decode step.
    - Isolated per-request KV cache states avoiding cross-sequence contamination.
    - Synchronous blocking batch completion and asynchronous streaming token iterators.
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
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id or getattr(tokenizer, "eos_token_id", None)

        self.batch_manager = TGIRequestBatch(max_batch_size=max_batch_size, max_seq_len=max_seq_len)

        # Operational metrics tracking
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

            inp = slot.input_ids.unsqueeze(0).to(self.device)  # [1, L]
            try:
                outputs = self.model(inp, use_cache=True)
                logits = outputs.logits[:, -1, :] if hasattr(outputs, "logits") else outputs[:, -1, :]
                pkv = getattr(outputs, "past_key_values", None)
            except TypeError:
                outputs = self.model(inp)
                logits = outputs.logits[:, -1, :] if hasattr(outputs, "logits") else outputs[:, -1, :]
                pkv = None

            slot.past_kv = pkv
            next_tok = self._sample_token(logits, slot.temperature, slot.top_k)
            slot.generated_ids.append(next_tok)
            self._total_tokens_generated += 1
            text_piece = self._decode_token(next_tok)

            is_stop = self.check_stop(slot, next_tok)
            if is_stop:
                slot.finished = True
                slot.finish_reason = "stop"
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

        # 1. Ensure un-prefilled requests complete prompt ingestion
        needs_prefill = [
            r_id for r_id in active_ids
            if r_id in self.batch_manager.active_slots
            and self.batch_manager.active_slots[r_id].past_kv is None
            and not self.batch_manager.active_slots[r_id].finished
        ]
        if needs_prefill:
            self.prefill(needs_prefill)

        # 2. Decode next token for all active, prefilled sequences
        for req_id in active_ids:
            slot = self.batch_manager.active_slots.get(req_id)
            if not slot or slot.finished:
                continue

            last_tok = slot.generated_ids[-1] if slot.generated_ids else int(slot.input_ids[-1].item())
            inp = torch.tensor([[last_tok]], dtype=torch.long, device=self.device)

            if slot.past_kv is not None:
                outputs = self.model(inp, past_key_values=slot.past_kv, use_cache=True)
                logits = outputs.logits[:, -1, :] if hasattr(outputs, "logits") else outputs[:, -1, :]
                slot.past_kv = getattr(outputs, "past_key_values", None)
            else:
                full_context = torch.cat([slot.input_ids, torch.tensor(slot.generated_ids, device=self.device)])
                outputs = self.model(full_context.unsqueeze(0))
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
            req_id = f"req_{int(time.time() * 1000)}_{i}"
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
        Returns a FastAPI application instance exposing TGI-compatible endpoints:
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

        app = FastAPI(title="NeuralOps TGI Serving Adapter", version="1.3.7")

        @app.get("/health")
        def health() -> Dict[str, str]:
            return {"status": "ok"}

        @app.get("/info")
        def info() -> Dict[str, Any]:
            return self.stats()

        @app.post("/generate")
        async def generate(request: Request) -> Dict[str, Any]:
            data = await request.json()
            prompt = data.get("inputs", "")
            params = data.get("parameters", {})
            max_new = params.get("max_new_tokens", 16)
            completions = self.continuous_batching_generate([prompt], max_new_tokens=max_new)
            return {"generated_text": completions[0]}

        @app.post("/generate_stream")
        async def generate_stream(request: Request) -> StreamingResponse:
            data = await request.json()
            prompt = data.get("inputs", "")
            params = data.get("parameters", {})
            max_new = params.get("max_new_tokens", 16)

            def event_generator():
                for tok, text in self.streaming_generate(prompt, max_new_tokens=max_new):
                    yield f"data: {{\"token\": {tok}, \"text\": \"{text}\"}}\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        return app


__all__ = [
    "TGIRequest",
    "TGIRequestSlot",
    "TGIRequestBatch",
    "TGIStreamingIterator",
    "TGIServingAdapter",
]
