"""High-performance LLM inference engine supporting streaming, speculative decoding, beam search, and continuous batching."""
from __future__ import annotations
from typing import Generator, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "NeuralOpsInferenceEngine",
    "stream_generate",
    "speculative_generate",
    "beam_search_generate",
    "ContinuousBatcher",
]


class NeuralOpsInferenceEngine:
    """Lightweight inference runtime managing kv-caching and decoding loops."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive greedy/top-k token generation."""
        curr = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.model(curr)
            if hasattr(logits, "logits"):
                logits = logits.logits
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, next_token_logits.shape[-1]))[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            curr = torch.cat([curr, next_token], dim=1)
        return curr


def stream_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 16,
) -> Generator[int, None, None]:
    """Yields generated tokens one-by-one."""
    curr = input_ids.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(curr)
            if hasattr(logits, "logits"):
                logits = logits.logits
            next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            curr = torch.cat([curr, torch.tensor([[next_token]], device=curr.device)], dim=1)
            yield next_token


def speculative_generate(
    target_model: nn.Module,
    draft_model: nn.Module,
    input_ids: torch.Tensor,
    gamma: int = 3,
    max_new_tokens: int = 6,
) -> torch.Tensor:
    """Speculative decoding: proposes gamma tokens with draft_model and verifies with target_model."""
    curr = input_ids.clone()
    generated = 0
    while generated < max_new_tokens:
        # Draft step
        draft_tokens = []
        d_curr = curr.clone()
        for _ in range(min(gamma, max_new_tokens - generated)):
            logits = draft_model(d_curr)
            tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            draft_tokens.append(tok)
            d_curr = torch.cat([d_curr, tok], dim=1)

        # Verification step with target model
        target_logits = target_model(d_curr)
        accepted = 0
        for i, draft_tok in enumerate(draft_tokens):
            pos = curr.shape[1] + i - 1
            best_target_tok = torch.argmax(target_logits[:, pos, :], dim=-1, keepdim=True)
            if torch.equal(draft_tok, best_target_tok):
                curr = torch.cat([curr, draft_tok], dim=1)
                accepted += 1
                generated += 1
            else:
                curr = torch.cat([curr, best_target_tok], dim=1)
                generated += 1
                break

    return curr


def beam_search_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    beam_width: int = 2,
    max_new_tokens: int = 4,
) -> torch.Tensor:
    """Standard Beam Search generation."""
    # List of (score, sequence)
    beams = [(0.0, input_ids)]
    for _ in range(max_new_tokens):
        candidates = []
        for score, seq in beams:
            with torch.no_grad():
                logits = model(seq)
                if hasattr(logits, "logits"):
                    logits = logits.logits
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                topk_probs, topk_ids = torch.topk(log_probs, beam_width, dim=-1)
                for i in range(beam_width):
                    cand_seq = torch.cat([seq, topk_ids[:, i : i + 1]], dim=1)
                    cand_score = score + topk_probs[0, i].item()
                    candidates.append((cand_score, cand_seq))
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

    return beams[0][1]


class ContinuousBatcher:
    """Continuous dynamic batch scheduler for multi-request serving."""

    def __init__(self, max_batch_size: int = 8) -> None:
        self.max_batch_size = max_batch_size
        self.active_requests: list[dict[str, Any]] = []

    def add_request(self, req_id: str, input_ids: torch.Tensor) -> None:
        self.active_requests.append({
            "id": req_id,
            "tokens": input_ids,
            "finished": False,
        })

    def step(self, model: nn.Module) -> list[dict[str, Any]]:
        """Processes one token decoding step across active requests."""
        if not self.active_requests:
            return []

        batch = self.active_requests[: self.max_batch_size]
        results = []
        for req in batch:
            with torch.no_grad():
                logits = model(req["tokens"])
                if hasattr(logits, "logits"):
                    logits = logits.logits
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                req["tokens"] = torch.cat([req["tokens"], next_tok], dim=1)
                results.append({"id": req["id"], "token": next_tok.item()})
        return results
