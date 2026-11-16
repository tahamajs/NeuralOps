# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Inference pipeline and OpenAI serving: TextGenerationPipeline, pipeline, serve.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.API")


class TextGenerationPipeline:
    """High-level pipeline wrapping autoregressive text generation with clean decoded output."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any = None,
        device: Any = None,
        **kwargs: Any,
    ) -> None:
        self.task = "text-generation"
        self.device = device
        if device is not None:
            try:
                model = model.to(device)
            except Exception as e:
                logger.warning("Failed to move model to target device '%s': %s", device, e)
        self.model = model
        self.tokenizer = tokenizer
        self.default_kwargs = dict(kwargs)

    def __call__(
        self,
        inputs: Any = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        prompt_input = inputs
        if prompt_input is None:
            prompt_input = kwargs.pop("prompt", None)
        if prompt_input is None:
            prompt_input = kwargs.pop("text", None)
        if prompt_input is None:
            prompt_input = kwargs.pop("inputs", None)
        if prompt_input is None:
            raise ValueError("No input text or prompt provided to pipeline.")

        tok = self.tokenizer
        if tok is None:
            tok = getattr(self.model, "tokenizer", None)
        if tok is None:
            config = getattr(self.model, "config", None)
            model_name = getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", None)
            if model_name:
                try:
                    from transformers import AutoTokenizer

                    tok = AutoTokenizer.from_pretrained(model_name)
                    self.tokenizer = tok
                except Exception as e:
                    logger.warning("Failed to automatically load tokenizer for '%s': %s", model_name, e)

        # Handle torch.Tensor inputs directly
        if isinstance(prompt_input, torch.Tensor):
            input_ids = prompt_input
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            target_dev = self._get_device()
            if target_dev is not None:
                try:
                    input_ids = input_ids.to(target_dev)
                except Exception as e:
                    logger.warning("Failed to transfer input_ids tensor to device '%s': %s", target_dev, e)
            output_ids = self._generate_tokens(input_ids, max_length, max_new_tokens, **kwargs)
            if tok is not None:
                decoded = self._decode_tokens(tok, output_ids, **kwargs)
                return [{"generated_text": text} for text in decoded]
            return [{"generated_token_ids": output_ids}]

        # Handle string or list of strings
        if isinstance(prompt_input, str):
            prompts = [prompt_input]
            is_single = True
        elif isinstance(prompt_input, (list, tuple)):
            prompts = [str(p) for p in prompt_input]
            is_single = False
        else:
            prompts = [str(prompt_input)]
            is_single = True

        if tok is None:
            raise ValueError("A tokenizer must be provided to encode string inputs for 'text-generation' pipeline.")

        input_ids, attention_mask = self._encode_prompts(tok, prompts)
        target_dev = self._get_device()
        if target_dev is not None:
            try:
                input_ids = input_ids.to(target_dev)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(target_dev)
            except Exception as e:
                logger.warning("Failed to transfer input batch to device '%s': %s", target_dev, e)

        output_ids = self._generate_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        decoded = self._decode_tokens(tok, output_ids, **kwargs)

        if is_single:
            return [{"generated_text": decoded[0]}]
        return [[{"generated_text": text}] for text in decoded]

    def _get_device(self) -> Any:
        if self.device is not None:
            return self.device
        try:
            return next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device("cpu")

    def _encode_prompts(self, tok: Any, prompts: List[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        encoded = None
        if callable(tok):
            try:
                encoded = tok(prompts, return_tensors="pt", padding=True, truncation=True)
            except TypeError:
                try:
                    encoded = tok(prompts, return_tensors="pt", padding=True)
                except TypeError:
                    try:
                        encoded = tok(prompts, return_tensors="pt")
                    except TypeError:
                        encoded = tok(prompts)
        elif hasattr(tok, "encode"):
            ids_list = [tok.encode(p) for p in prompts]
            if isinstance(ids_list[0], torch.Tensor):
                encoded = torch.stack(ids_list)
            else:
                encoded = torch.tensor(ids_list, dtype=torch.long)

        if isinstance(encoded, dict):
            input_ids = encoded.get("input_ids")
            attention_mask = encoded.get("attention_mask", None)
        elif isinstance(encoded, torch.Tensor):
            input_ids = encoded
            attention_mask = None
        elif isinstance(encoded, (list, tuple)):
            input_ids = torch.tensor(encoded, dtype=torch.long)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            attention_mask = None
        else:
            raise ValueError(f"Could not encode prompts with tokenizer: unexpected type {type(encoded)}")

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        return input_ids, attention_mask

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        call_kwargs = dict(self.default_kwargs)
        call_kwargs.update(kwargs)
        if max_length is not None:
            call_kwargs["max_length"] = max_length
        if max_new_tokens is not None:
            call_kwargs["max_new_tokens"] = max_new_tokens

        resolved_tokens = call_kwargs.get("max_new_tokens")
        if resolved_tokens is None:
            if "max_length" in call_kwargs and call_kwargs["max_length"] is not None:
                resolved_tokens = max(1, call_kwargs["max_length"] - input_ids.shape[1])
            else:
                resolved_tokens = 20

        if hasattr(self.model, "generate") and callable(self.model.generate):
            gen_args = dict(call_kwargs)
            gen_args.setdefault("max_new_tokens", resolved_tokens)
            if attention_mask is not None and "attention_mask" not in gen_args:
                gen_args["attention_mask"] = attention_mask
            try:
                return self.model.generate(input_ids=input_ids, **gen_args)
            except TypeError:
                return self.model.generate(input_ids)

        # Fallback to NeuralOpsInferenceEngine
        from neuralops.inference.engine import NeuralOpsInferenceEngine

        engine = NeuralOpsInferenceEngine(self.model)
        return engine.generate(
            input_ids=input_ids,
            max_new_tokens=resolved_tokens,
            temperature=call_kwargs.get("temperature", 1.0),
            top_k=call_kwargs.get("top_k", None),
            top_p=call_kwargs.get("top_p", None),
            repetition_penalty=call_kwargs.get("repetition_penalty", 1.0),
        )

    def _decode_tokens(self, tok: Any, output_ids: torch.Tensor, **kwargs: Any) -> List[str]:
        if isinstance(output_ids, torch.Tensor):
            output_ids_cpu = output_ids.detach().cpu()
        else:
            output_ids_cpu = output_ids

        decoded: List[str] = []
        if hasattr(tok, "batch_decode"):
            try:
                decoded = tok.batch_decode(output_ids_cpu, skip_special_tokens=True)
            except Exception as e_batch:
                try:
                    decoded = tok.batch_decode(output_ids_cpu)
                except Exception as e_batch2:
                    logger.warning(
                        "Tokenizer batch_decode failed (skip_special: %s, unskipped: %s); falling back to per-sequence decode",
                        e_batch,
                        e_batch2,
                    )
        if not decoded and hasattr(tok, "decode"):
            for seq in output_ids_cpu:
                try:
                    decoded.append(tok.decode(seq, skip_special_tokens=True))
                except Exception as e_skip:
                    try:
                        decoded.append(tok.decode(seq))
                    except Exception as e_seq:
                        logger.error(
                            "Tokenizer failed to decode sequence %s: %s (skip_special attempt: %s)",
                            seq,
                            e_seq,
                            e_skip,
                        )
                        raise RuntimeError(f"Tokenizer failed to decode token IDs: {e_seq}") from e_seq
        if not decoded:
            raise RuntimeError("Tokenizer failed to decode generated tokens; no valid decoded output produced.")

        clean_spaces = kwargs.get("clean_up_tokenization_spaces", True)
        results: List[str] = []
        for text in decoded:
            clean = text.strip() if clean_spaces else text
            results.append(clean)
        return results

    predict = __call__
    generate = __call__


def pipeline(
    task: str,
    model: nn.Module,
    tokenizer: Any = None,
    device: Any = None,
    **kwargs: Any,
) -> Any:
    """Create a high-level NeuralOps pipeline for model inference.

    Args:
        task: The task to perform. Currently supported: 'text-generation'.
        model: PyTorch or HuggingFace neural network module.
        tokenizer: Optional tokenizer for encoding inputs and decoding generated tokens.
        device: Target execution device (e.g. 'cpu', 'cuda', 'cuda:0', torch.device).
        **kwargs: Additional configuration parameters passed to the pipeline or generation runtime.

    Returns:
        A callable Pipeline instance, or generated output if prompt/inputs were provided in kwargs.
    """
    normalized_task = task.lower().strip()
    if normalized_task not in ("text-generation", "text_generation"):
        raise ValueError(f"Unsupported pipeline task '{task}'. Currently supported tasks: ['text-generation'].")

    pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=device, **kwargs)

    # If prompt/inputs was passed directly in kwargs, execute immediately
    if "prompt" in kwargs:
        call_kwargs = dict(kwargs)
        prompt = call_kwargs.pop("prompt")
        return pipe(prompt, **call_kwargs)
    if "inputs" in kwargs:
        call_kwargs = dict(kwargs)
        inputs = call_kwargs.pop("inputs")
        return pipe(inputs, **call_kwargs)

    return pipe


def serve(
    model: Union[str, nn.Module],
    port: int = 8000,
    host: str = "127.0.0.1",
    backend: str = "auto",
    kv_cache_budget_gb: float = 4.0,
    api_key: Optional[str] = None,
    rate_limit_rpm: int = 60,
    **kwargs: Any,
) -> Any:
    """Launches an OpenAI-compatible HTTP serving endpoint for the model.

    Canonical Serving Note (F-60):
        The serving engine is powered canonically by
        `neuralops.integrations.serving.create_openai_serving_app`
        (re-exported as `neuralops.serving.create_openai_serving_app`),
        providing OpenAI-compatible chat completions and generation endpoints.

    Parameters:
        model: PyTorch model instance or Hugging Face model identifier string.
        port: HTTP server port (default: 8000).
        host: Host binding address (default: '127.0.0.1').
        backend: Serving engine ('vllm', 'sglang', or 'neuralops_native').
        kv_cache_budget_gb: Maximum VRAM allocation for paged KV-cache.
        api_key: Optional API key for client authentication. Defaults to NEURALOPS_API_KEY.
        rate_limit_rpm: Maximum requests per minute per IP address (default: 60).
    """
    resolved_api_key = api_key or os.environ.get("NEURALOPS_API_KEY")
    if host not in ("127.0.0.1", "localhost") and not resolved_api_key:
        raise ValueError(
            "Refusing to bind serving to non-localhost address without an API key configured (set NEURALOPS_API_KEY)."
        )

    if isinstance(model, str):
        from transformers import AutoModelForCausalLM

        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16 if resolved_device.type == "cuda" else torch.float32,
        ).to(resolved_device)

    from neuralops.integrations.serving import create_openai_serving_app

    app = create_openai_serving_app(
        model,
        backend=backend,
        kv_cache_budget_gb=kv_cache_budget_gb,
        api_key=resolved_api_key,
        rate_limit_rpm=rate_limit_rpm,
        **kwargs,
    )
    import uvicorn

    return uvicorn.run(app, host=host, port=port)
