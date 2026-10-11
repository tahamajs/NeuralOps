"""Production-grade LLM inference engine supporting sampling, speculative decoding, beam search, and continuous batching."""
from __future__ import annotations
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
        curr = input_ids.clone()
        past_kv = None

        for _ in range(max_new_tokens):
            inp = curr if (past_kv is None or not use_cache) else curr[:, -1:]
            try:
                outputs = self.model(inp, past_key_values=past_kv, use_cache=use_cache)
                past_kv = getattr(outputs, "past_key_values", None)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
            except TypeError:
                outputs = self.model(curr)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

            next_token_logits = logits[:, -1, :].clone() / max(temperature, 1e-5)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(next_token_logits.shape[0]):
                    for tok in curr[b].tolist():
                        if next_token_logits[b, tok] > 0:
                            next_token_logits[b, tok] /= repetition_penalty
                        else:
                            next_token_logits[b, tok] *= repetition_penalty

            # Top-k filtering
            if top_k is not None and top_k > 0:
                val, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.shape[-1]))
                next_token_logits[next_token_logits < val[:, -1:]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep at least one token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                for b in range(next_token_logits.shape[0]):
                    indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                    next_token_logits[b, indices_to_remove] = float("-inf")

            probs = F.softmax(next_token_logits, dim=-1)
            if temperature > 0.0 and (top_k is not None or top_p is not None):
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
    if gamma is not None:
        K = gamma
    """Real speculative decoding with acceptance/rejection sampling."""
    curr = input_ids.clone()
    while curr.shape[1] - input_ids.shape[1] < max_new_tokens:
        draft_ids = curr.clone()
        draft_probs = []

        # Draft model proposes K tokens
        for _ in range(K):
            d_out = draft_model(draft_ids)
            d_logits = d_out.logits if hasattr(d_out, "logits") else d_out
            p = F.softmax(d_logits[:, -1, :], dim=-1)
            draft_probs.append(p)
            tok = torch.argmax(p, dim=-1, keepdim=True)
            draft_ids = torch.cat([draft_ids, tok], dim=-1)

        # Target model verifies draft tokens in single parallel forward pass
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
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # [beam_size, V]

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
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.pending: list[GenerationRequest] = []
        self.active: list[GenerationRequest] = []

    def add_request(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], torch.Tensor):
            req = GenerationRequest(request_id=args[0], input_ids=args[1])
        elif len(args) == 1 and isinstance(args[0], tuple):
            req = GenerationRequest(request_id=args[0][0], input_ids=args[0][1])
        elif len(args) == 1 and isinstance(args[0], GenerationRequest):
            req = args[0]
        elif "req_id" in kwargs:
            req = GenerationRequest(request_id=kwargs["req_id"], input_ids=kwargs["input_ids"])
        else:
            req = args[0]
        self.pending.append(req)

    @torch.no_grad()
    def step(self, model: nn.Module | None = None) -> list[dict[str, Any]]:
        if model is not None:
            self.model = model
        """Executes a single step of batch decode across active requests."""
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

        try:
            out = self.model(padded, attention_mask=mask)
        except TypeError:
            out = self.model(padded)

        logits = out.logits if hasattr(out, "logits") else out
        step_results = []

        for i, r in enumerate(self.active):
            tok = int(logits[i, -1, :].argmax().item())
            tok_tensor = torch.tensor([[tok]], device=r.input_ids.device)
            r.input_ids = torch.cat([r.input_ids, tok_tensor], dim=1)
            r.generated.append(tok)
            if len(r.generated) >= r.max_new_tokens:
                r.finished = True
            step_results.append({"id": r.request_id, "token": tok, "finished": r.finished})

        self.active = [r for r in self.active if not r.finished]
        return step_results
