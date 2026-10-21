#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Inference Serving Adapters

Integrates NeuralOps models and attention layers with vLLM and Text Generation Inference (TGI).
"""

from __future__ import annotations

import json
import math
from typing import Any, List, Optional
import torch
import torch.nn as nn

try:
    import vllm
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

try:
    import text_generation_server
    HAS_TGI = True
except ImportError:
    HAS_TGI = False


class VLLMAttentionAdapter:
    """Adapts NeuralOps attention modules for vLLM block-manager and forward runners."""

    def __init__(self, attention_type: str, head_dim: int, num_heads: int) -> None:
        self.attention_type = attention_type
        self.head_dim = head_dim
        self.num_heads = num_heads

    @staticmethod
    def is_available() -> bool:
        """Check if vLLM package is available."""
        return HAS_VLLM

    def register(self, backend_cls: type) -> bool:
        """Register a concrete vLLM AttentionBackend implementation."""
        if not HAS_VLLM:
            raise RuntimeError(
                "vLLM is not installed. Please install neuralops[serving] or run: pip install vllm"
            )
        backend_name = getattr(backend_cls, "get_name", lambda: None)()
        if not backend_name:
            raise ValueError("backend_cls must implement get_name() from vLLM's AttentionBackend contract")
        from neuralops.export.serving import VLLMAdapter

        return VLLMAdapter.register_attention_backend(backend_name, backend_cls)

    def format_metadata(self, block_tables: List[List[int]], context_lens: List[int]) -> dict[str, Any]:
        """Formats sequence metadata into vLLM engine-compatible structures."""
        return {
            "block_tables": block_tables,
            "context_lens": context_lens,
            "max_context_len": max(context_lens) if context_lens else 0,
        }

    def paged_attention_forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int = 16,
    ) -> torch.Tensor:
        """Execute paged attention forward pass over block_tables and KV-cache blocks."""
        if query.ndim != 3:
            raise ValueError("query must have shape [batch, heads, head_dim]")
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")

        from neuralops.core.triton_kernels.paged_attention import paged_attention_kernel
        return paged_attention_kernel(
            q=query,
            k_cache=key_cache,
            v_cache=value_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            block_size=block_size,
        )


class TGIAttentionAdapter:
    """Adapts attention modules for Text Generation Inference (TGI) batch execution."""

    def __init__(self, max_batch_size: int = 32) -> None:
        self.max_batch_size = max_batch_size

    @staticmethod
    def is_available() -> bool:
        """Check if TGI server is available."""
        return HAS_TGI

    def prepare_batch(self, batch_inputs: List[Any]) -> List[Any]:
        """Truncates or batches inputs to match TGI runtime batch boundaries."""
        return batch_inputs[: self.max_batch_size]


class SGLangRadixLinearCacheAdapter:
    """
    Bridges NeuralOps linear/recurrent state caching with SGLang RadixAttention interface.
    Supports tree-structured prefix state matching, reference locking, and LRU eviction.
    """

    def __init__(self, max_cached_nodes: int = 4096) -> None:
        if not isinstance(max_cached_nodes, int) or isinstance(max_cached_nodes, bool) or max_cached_nodes <= 0:
            raise ValueError("max_cached_nodes must be a positive integer")
        self.max_cached_nodes = max_cached_nodes
        self.tree: dict[tuple[int, ...], dict[str, Any]] = {}
        self.access_counter = 0

    def match_prefix(self, token_ids: list[int] | tuple[int, ...]) -> tuple[int, Optional[Any]]:
        """Find the longest cached prefix using tuple-key lookups (O(L²) slicing work)."""
        token_tuple = tuple(token_ids)
        best_match_len = 0
        best_state = None

        for length in range(len(token_tuple), 0, -1):
            prefix = token_tuple[:length]
            if prefix in self.tree:
                self.access_counter += 1
                self.tree[prefix]["last_accessed"] = self.access_counter
                return length, self.tree[prefix]["state"]

        return 0, None

    def cache_finished_req(self, token_ids: list[int] | tuple[int, ...], final_state: Any) -> None:
        """Caches state of a completed sequence, evicting cold unlocked nodes if capacity is reached."""
        prefix = tuple(token_ids)
        self.access_counter += 1
        if prefix in self.tree:
            self.tree[prefix]["state"] = final_state
            self.tree[prefix]["last_accessed"] = self.access_counter
        else:
            self._ensure_capacity()
            if len(self.tree) >= self.max_cached_nodes:
                raise RuntimeError("Prefix cache is full and all entries are locked")
            self.tree[prefix] = {
                "state": final_state,
                "ref_count": 0,
                "last_accessed": self.access_counter,
            }

    def cache_unfinished_req(self, token_ids: list[int] | tuple[int, ...], current_state: Any) -> None:
        """Locks and caches intermediate state during prefill/rollouts."""
        prefix = tuple(token_ids)
        self.cache_finished_req(prefix, current_state)
        self.inc_lock_ref(prefix)

    def inc_lock_ref(self, token_ids: list[int] | tuple[int, ...]) -> None:
        prefix = tuple(token_ids)
        if prefix in self.tree:
            self.tree[prefix]["ref_count"] += 1

    def dec_lock_ref(self, token_ids: list[int] | tuple[int, ...]) -> None:
        prefix = tuple(token_ids)
        if prefix in self.tree:
            self.tree[prefix]["ref_count"] = max(0, self.tree[prefix]["ref_count"] - 1)

    def evict(self, num_nodes: int = 1) -> int:
        """Evicts unreferenced nodes according to LRU policy."""
        if not isinstance(num_nodes, int) or isinstance(num_nodes, bool) or num_nodes < 0:
            raise ValueError("num_nodes must be a non-negative integer")
        evicted = 0
        unlocked = [
            (prefix, node["last_accessed"])
            for prefix, node in self.tree.items()
            if node["ref_count"] == 0
        ]
        unlocked.sort(key=lambda x: x[1])

        for prefix, _ in unlocked[:num_nodes]:
            del self.tree[prefix]
            evicted += 1
        return evicted

    def _ensure_capacity(self) -> None:
        if len(self.tree) >= self.max_cached_nodes:
            self.evict(max(1, len(self.tree) - self.max_cached_nodes + 1))


def create_openai_serving_app(
    model: nn.Module | None = None,
    tokenizer: Any = None,
    model_name: str = "neuralops-default",
) -> Any:
    """Create an OpenAI-shaped API backed by the supplied model and tokenizer.

    This lightweight adapter performs ordinary Hugging Face ``generate`` calls;
    it does not implement continuous batching or vLLM's streaming scheduler.
    """
    try:
        from fastapi import FastAPI, Request, HTTPException
        from fastapi.responses import StreamingResponse
    except ImportError:
        raise ImportError(
            "FastAPI is required for OpenAI-compatible serving. "
            "Install with: pip install fastapi uvicorn"
        )

    import time
    import uuid

    app = FastAPI(title="NeuralOps Serving Engine", version="1.4.2")

    @app.get("/health")
    def health():
        ready = model is not None and tokenizer is not None
        return {"status": "ok" if ready else "not_ready", "ready": ready, "backend": "neuralops"}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "neuralops",
                }
            ],
        }

    @app.post("/v1/completions")
    async def create_completion(request: Request):
        from starlette.concurrency import run_in_threadpool

        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        prompt = body.get("prompt", "")
        max_tokens, temperature, stream = _read_generation_options(body)
        _validate_generation_request(model, tokenizer, prompt, max_tokens, temperature)
        generated_text, prompt_tokens, completion_tokens = await run_in_threadpool(
            _generate_text, model, tokenizer, prompt, max_tokens, temperature
        )

        req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if stream:
            async def event_generator():
                chunk = {
                    "id": req_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{"text": generated_text, "index": 0, "logprobs": None, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        return {
            "id": req_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "text": generated_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: Request):
        from starlette.concurrency import run_in_threadpool

        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        messages = body.get("messages", [])
        max_tokens, temperature, stream = _read_generation_options(body)
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")
        if any(not isinstance(message, dict) or not isinstance(message.get("role"), str) for message in messages):
            raise HTTPException(status_code=400, detail="each message must contain a string role")

        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        prompt = _format_chat_prompt(tokenizer, messages)
        _validate_generation_request(model, tokenizer, prompt, max_tokens, temperature)
        generated_text, prompt_tokens, completion_tokens = await run_in_threadpool(
            _generate_text, model, tokenizer, prompt, max_tokens, temperature
        )

        if stream:
            async def event_generator():
                chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": generated_text}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        return {
            "id": req_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def _read_generation_options(body: dict[str, Any]) -> tuple[int, float, bool]:
    """Parse OpenAI-style generation options and return HTTP 400 on malformed values."""
    from fastapi import HTTPException

    try:
        max_tokens = int(body.get("max_tokens", 16))
        temperature = float(body.get("temperature", 1.0))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="max_tokens and temperature must be numeric") from exc
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise HTTPException(status_code=400, detail="stream must be a boolean")
    return max_tokens, temperature, stream


def _validate_generation_request(
    model: nn.Module | None,
    tokenizer: Any,
    prompt: Any,
    max_tokens: int,
    temperature: float,
) -> None:
    """Reject invalid or unserviceable generation requests before model execution."""
    from fastapi import HTTPException

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="A model and tokenizer must be loaded to serve requests")
    if not isinstance(prompt, str):
        raise HTTPException(status_code=400, detail="prompt must be a string")
    if max_tokens <= 0:
        raise HTTPException(status_code=400, detail="max_tokens must be positive")
    if not math.isfinite(temperature) or temperature < 0:
        raise HTTPException(status_code=400, detail="temperature must be a finite, non-negative number")


def _format_chat_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    """Render chat messages with the tokenizer's template or a plain role-tagged prompt."""
    if any(not isinstance(message.get("content", ""), str) for message in messages):
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="message content must be a string")
    if callable(getattr(tokenizer, "apply_chat_template", None)):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{message['role']}: {message.get('content', '')}" for message in messages) + "\nassistant:"


def _generate_text(
    model: nn.Module | None,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, int, int]:
    """Run one causal generation request and return decoded text and token counts."""
    from fastapi import HTTPException

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="A model and tokenizer must be loaded to serve requests")
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise RuntimeError("Tokenizer must return input_ids with shape [batch, sequence]")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = input_ids.device
    except AttributeError:
        device = input_ids.device
    encoded = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in encoded.items()}
    generation_kwargs: dict[str, Any] = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    model.eval()
    with torch.inference_mode():
        sequences = model.generate(**encoded, **generation_kwargs)
    if hasattr(sequences, "sequences"):
        sequences = sequences.sequences
    if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
        raise RuntimeError("Model.generate must return token IDs with shape [batch, sequence]")
    generated = sequences[0, input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    prompt_count = int(encoded.get("attention_mask", torch.ones_like(input_ids)).sum().item())
    return text, prompt_count, int(generated.numel())


VLLMAdapter = VLLMAttentionAdapter
TGIAdapter = TGIAttentionAdapter

__all__ = [
    "VLLMAttentionAdapter",
    "TGIAttentionAdapter",
    "VLLMAdapter",
    "TGIAdapter",
    "SGLangRadixLinearCacheAdapter",
    "create_openai_serving_app",
    "HAS_VLLM",
    "HAS_TGI",
]
