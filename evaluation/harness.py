#!/usr/bin/env python3
"""
lm-evaluation-harness integration and CLI evaluation runner for NeuralOps models.
Supports evaluating standard zero-shot tasks (wikitext, lambada, piqa, hellaswag,
winogrande, arc_easy, arc_challenge, boolq, sciq, copa, openbookqa) natively.
Registers 'neuralops' and 'fla' model wrappers into lm_eval.

Author: Taha Majlesi
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Framework paths
HERE = Path(__file__).resolve().parent
CODES = HERE.parent
ROOT = CODES.parent
for p in [str(CODES), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation.eval_patcher import apply_module_to_model

logger = logging.getLogger("evaluation.harness")

# Graceful detection of lm-eval
try:
    import lm_eval
    import lm_eval.api.model
    import lm_eval.api.registry
    from lm_eval.api.instance import Instance
    from lm_eval import evaluator
    BaseLM = lm_eval.api.model.LM
    register_model = lm_eval.api.registry.register_model
    HAS_LM_EVAL = True
except (ImportError, AttributeError, ModuleNotFoundError):
    try:
        from lm_eval.base import LM as BaseLM
        HAS_LM_EVAL = True
        register_model = lambda *names: lambda cls: cls
    except (ImportError, AttributeError, ModuleNotFoundError):
        class BaseLM:
            """Baselm definition."""
            pass
        register_model = lambda *names: lambda cls: cls
        HAS_LM_EVAL = False


@register_model("neuralops", "fla")
class NeuralOpsLMWrapper(BaseLM):
    """
    LM-Evaluation-Harness Model Wrapper for NeuralOps and FLA-compatible models.
    Supports causal loglikelihood evaluation, rolling loglikelihood, and generate_until.
    """

    def __init__(
        self,
        pretrained: Union[str, nn.Module] = "gpt2",
        module_name: Optional[str] = None,
        module_type: str = "attention",
        tokenizer: Optional[Any] = None,
        device: Union[str, torch.device] = "cuda",
        dtype: Union[str, torch.dtype] = "float16",
        batch_size: Union[int, str] = 1,
        max_length: Optional[int] = 2048,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> None:
        """Execute init routine."""
        super().__init__()
        
        self._device = torch.device(device if (torch.cuda.is_available() and device != "cpu") else "cpu")
        
        if isinstance(dtype, str):
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            self._dtype = dtype_map.get(dtype.lower(), torch.float16 if self._device.type == "cuda" else torch.float32)
        else:
            self._dtype = dtype

        # Model initialization
        if isinstance(pretrained, str):
            self.model_name_or_path = pretrained
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                torch_dtype=self._dtype,
                trust_remote_code=trust_remote_code,
            )
        else:
            self.model = pretrained
            self.model_name_or_path = getattr(pretrained, "name_or_path", "neuralops_model")

        # Apply NeuralOps attention or compression module if specified
        self.module_name = module_name
        self.module_type = module_type
        if module_name and module_name.lower() not in ["none", "null", "baseline"]:
            logger.info(f"Patching model with NeuralOps {module_type}: {module_name}")
            self.model, _ = apply_module_to_model(
                self.model,
                module_name=module_name,
                module_type=module_type,
            )

        self.model.to(self._device)
        self.model.eval()

        # Tokenizer setup
        if tokenizer is None:
            if isinstance(pretrained, str):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    pretrained,
                    trust_remote_code=trust_remote_code,
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        elif isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=trust_remote_code)
        else:
            self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                self.model.resize_token_embeddings(len(self.tokenizer))

        self.tokenizer.padding_side = "left"

        self._batch_size = int(batch_size) if str(batch_size).isdigit() else 1
        self._max_length = max_length or getattr(self.model.config, "max_position_embeddings", 2048)

    @property
    def eot_token_id(self) -> int:
        """Execute eot token id routine."""
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        """Execute max length routine."""
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        """Execute max gen toks routine."""
        return 256

    @property
    def batch_size(self) -> int:
        """Execute batch size routine."""
        return self._batch_size

    @property
    def device(self) -> torch.device:
        """Execute device routine."""
        return self._device

    def tok_encode(self, string: str, left_truncate_len: Optional[int] = None, add_special_tokens: bool = False) -> List[int]:
        """Execute tok encode routine."""
        enc = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            enc = enc[-left_truncate_len:]
        return enc

    def tok_decode(self, tokens: List[int]) -> str:
        """Execute tok decode routine."""
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Any], disable_tqdm: bool = False) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood for context -> continuation pairs.
        Returns list of (logprob, is_greedy).
        """
        results: List[Tuple[float, bool]] = []
        for req in requests:
            if hasattr(req, "args"):
                context, continuation = req.args
            else:
                context, continuation = req

            if not context:
                ctx_tokens = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else [0]
            else:
                ctx_tokens = self.tok_encode(context)

            cont_tokens = self.tok_encode(continuation)
            full_tokens = ctx_tokens + cont_tokens
            if len(full_tokens) > self.max_length:
                full_tokens = full_tokens[-self.max_length:]
                ctx_len = max(1, len(full_tokens) - len(cont_tokens))
            else:
                ctx_len = len(ctx_tokens)

            cont_len = len(full_tokens) - ctx_len
            if cont_len <= 0:
                results.append((0.0, True))
                continue

            inp = torch.tensor([full_tokens], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = self.model(inp).logits[0]  # [seq_len, vocab_size]
                shift_logits = logits[:-1, :]
                shift_targets = inp[0, 1:]
                
                # Slicing target continuation tokens
                cont_logits = shift_logits[ctx_len - 1:]
                cont_targets = shift_targets[ctx_len - 1:]

                log_probs = F.log_softmax(cont_logits, dim=-1)
                selected_log_probs = log_probs.gather(dim=-1, index=cont_targets.unsqueeze(-1)).squeeze(-1)
                sum_log_prob = float(selected_log_probs.sum().item())

                greedy_tokens = cont_logits.argmax(dim=-1)
                is_greedy = bool((greedy_tokens == cont_targets).all().item())

            results.append((sum_log_prob, is_greedy))
        return results

    def loglikelihood_rolling(self, requests: List[Any], disable_tqdm: bool = False) -> List[float]:
        """
        Compute perplexity / rolling loglikelihood for longer contexts.
        """
        results: List[float] = []
        for req in requests:
            string = req.args[0] if hasattr(req, "args") else req
            tokens = self.tok_encode(string)
            if not tokens:
                results.append(0.0)
                continue

            seq_len = len(tokens)
            stride = self.max_length // 2 if self.max_length > 1 else 1
            nlls: List[float] = []

            for i in range(0, seq_len, stride):
                begin = max(i + stride - self.max_length, 0)
                end = min(i + stride, seq_len)
                trg_len = end - i
                input_chunk = torch.tensor([tokens[begin:end]], dtype=torch.long, device=self.device)
                target_chunk = input_chunk.clone()
                target_chunk[:, :-trg_len] = -100

                with torch.no_grad():
                    logits = self.model(input_chunk).logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = target_chunk[..., 1:].contiguous()
                    loss_fn = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
                    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    nlls.append(loss.item())

                if end == seq_len:
                    break

            results.append(-float(sum(nlls)))
        return results

    def generate_until(self, requests: List[Any], disable_tqdm: bool = False) -> List[str]:
        """
        Greedy / sampling generation up to stopping criteria.
        """
        results: List[str] = []
        for req in requests:
            if hasattr(req, "args"):
                context, gen_kwargs = req.args
            else:
                context = req[0]
                gen_kwargs = req[1] if len(req) > 1 else {}

            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]

            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            inputs = self.tokenizer(context, return_tensors="pt").to(self.device)

            with torch.no_grad():
                gen_out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_gen,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=gen_kwargs.get("do_sample", False),
                    temperature=gen_kwargs.get("temperature", 1.0),
                )
                generated_tokens = gen_out[0][inputs.input_ids.shape[1]:]
                gen_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            for stop in until:
                if stop in gen_text:
                    gen_text = gen_text.split(stop)[0]

            results.append(gen_text)
        return results


def run_evaluation(
    model: Union[str, nn.Module] = "gpt2",
    module_name: Optional[str] = None,
    module_type: str = "attention",
    tasks: Optional[List[str]] = None,
    num_fewshot: int = 0,
    batch_size: int = 1,
    max_length: int = 2048,
    limit: Optional[int] = None,
    device: str = "cuda",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run zero-shot evaluation on lm-eval tasks using NeuralOpsLMWrapper.
    """
    tasks = tasks or ["wikitext", "piqa", "hellaswag"]
    lm = NeuralOpsLMWrapper(
        pretrained=model,
        module_name=module_name,
        module_type=module_type,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )

    if not HAS_LM_EVAL:
        logger.warning("lm_eval not installed. Returning dry-run evaluation stub.")
        return {"model": str(model), "module": module_name, "tasks": tasks, "status": "lm_eval_not_installed"}

    try:
        from lm_eval import simple_evaluate
        eval_results = simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            limit=limit,
        )
    except Exception as e:
        logger.error(f"Error executing lm_eval simple_evaluate: {e}")
        eval_results = {"error": str(e), "tasks": tasks}

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, default=str)
        logger.info(f"Saved evaluation results to {output_path}")

    return eval_results


def main() -> None:
    """Execute main routine."""
    parser = argparse.ArgumentParser(description="NeuralOps & FLA LM-Evaluation Harness Entrypoint")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model path or name")
    parser.add_argument("--module-name", type=str, default=None, help="NeuralOps attention or compressor module")
    parser.add_argument("--module-type", type=str, default="attention", choices=["attention", "compressor"])
    parser.add_argument("--tasks", nargs="+", default=["wikitext", "piqa", "hellaswag"], help="lm-eval tasks")
    parser.add_argument("--num-fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluation examples per task")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None, help="Path to save output JSON")
    args = parser.parse_args()

    results = run_evaluation(
        model=args.model,
        module_name=args.module_name,
        module_type=args.module_type,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_length=args.max_length,
        limit=args.limit,
        device=args.device,
        output_path=args.output,
    )
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
