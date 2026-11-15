#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SGLang Serving Plugin Entrypoint.

Provides plugin registration and offline mock serving runner for SGLang.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.integrations.sglang import get_sglang_radix_adapter, sglang_is_available
from neuralops.integrations.sglang.radix_recurrent import RadixRecurrentTree

logger = logging.getLogger("NeuralOps.SGLangPlugin")

__all__ = [
    "HeadlessSGLangRunner",
    "get_sglang_runner",
    "register_neuralops_sglang_backend",
]


class HeadlessSGLangRunner:
    """
    Offline headless mock runner for SGLang serving workloads in CI / local environments.

    Emulates SGLang request processing, Radix prefix caching, and linear attention
    forward passes without requiring live GPU cluster or SGLang C++ binaries.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tree: Optional[RadixRecurrentTree] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the HeadlessSGLangRunner.

        Args:
            model: Optional PyTorch model instance to execute forward passes.
            tree: Optional pre-configured RadixRecurrentTree for prefix caching.
            device: Execution device (e.g. 'cpu' or 'cuda').
            **kwargs: Extra arguments passed to internal components.
        """
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.model = model.to(self.device).eval() if model is not None else None
        self.tree = tree if tree is not None else RadixRecurrentTree()
        self.radix_adapter = get_sglang_radix_adapter(tree=self.tree, device=self.device)
        self.kwargs = kwargs
        self._is_initialized = True

    def generate(
        self,
        prompt: Union[str, List[int], torch.Tensor],
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate completion tokens offline using mock runtime or attached model.

        Args:
            prompt: String, list of token IDs, or integer tensor prompt.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (for API parity).
            **kwargs: Additional generation parameters.

        Returns:
            Dictionary containing generated tokens, text, status, and cached token count.
        """
        if isinstance(prompt, str):
            token_ids = [ord(c) % 256 for c in prompt]
            prompt_text = prompt
        elif isinstance(prompt, torch.Tensor):
            token_ids = prompt.flatten().tolist()
            prompt_text = "".join(chr(t % 128) for t in token_ids)
        else:
            token_ids = list(prompt)
            prompt_text = "".join(chr(t % 128) for t in token_ids)

        # Check Radix cache for prefix hit
        matched_len, cached_state = self.radix_adapter.match_prefix_state(token_ids)

        generated_tokens: List[int] = []
        cur_tokens = list(token_ids)

        for step in range(max_new_tokens):
            if self.model is not None:
                inp = torch.tensor([cur_tokens], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    logits = self.model(inp)
                    if hasattr(logits, "logits"):
                        logits = logits.logits
                    next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            else:
                # Deterministic synthetic token generation for offline CI verification
                next_token = (cur_tokens[-1] + 1) % 1000

            generated_tokens.append(next_token)
            cur_tokens.append(next_token)

        # Cache final sequence prefix state
        dummy_state = torch.ones((1, 4, 32, 32), dtype=torch.float32, device=self.device)
        self.radix_adapter.insert_prefix_state(cur_tokens, dummy_state)

        return {
            "status": "success",
            "prompt": prompt_text,
            "prompt_tokens": token_ids,
            "generated_tokens": generated_tokens,
            "output_tokens": cur_tokens,
            "cached_tokens": matched_len,
            "finish_reason": "length",
        }

    def execute_batch(
        self,
        requests: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Execute a batch of generation requests sequentially.

        Args:
            requests: List of request dictionaries containing 'prompt' and generation kwargs.

        Returns:
            List of generation result dictionaries.
        """
        results: List[Dict[str, Any]] = []
        for req in requests:
            prompt = req.get("prompt", "")
            max_new_tokens = req.get("max_new_tokens", 8)
            res = self.generate(prompt=prompt, max_new_tokens=max_new_tokens)
            results.append(res)
        return results

    def shutdown(self) -> None:
        """Release runner resources."""
        self._is_initialized = False


def register_neuralops_sglang_backend(
    config_class: Optional[type] = None,
    arch_names: Optional[list[str]] = None,
    **kwargs: Any,
) -> bool:
    """
    Register NeuralOps Linear Attention Backend within SGLang runtime.

    Args:
        config_class: Optional config class for architecture registration.
        arch_names: Optional architecture names.
        **kwargs: Additional registration options.

    Returns:
        True if registration succeeded, False otherwise.

    Raises:
        IntegrationNotImplementedError: If SGLang is not installed and offline mock is disabled.
    """
    if not sglang_is_available():
        if os.environ.get("NEURALOPS_MOCK_SERVING") == "1":
            logger.info("NEURALOPS_MOCK_SERVING=1 active; registered mock SGLang linear backend.")
            return True
        raise IntegrationNotImplementedError(
            "SGLang is not installed. To register NeuralOps linear attention plugins into SGLang, "
            "install SGLang with:\n"
            "    pip install 'sglang>=0.4.0'\n"
            "Or set NEURALOPS_MOCK_SERVING=1 for offline CI testing."
        )

    from neuralops.integrations.sglang_backend.registry import register

    return register(config_class=config_class, arch_names=arch_names, **kwargs)


def get_sglang_runner(
    model: Optional[nn.Module] = None,
    headless_mock: bool = False,
    device: Union[str, torch.device] = "cpu",
    **kwargs: Any,
) -> Any:
    """
    Retrieve or construct an SGLang serving runner.

    Args:
        model: Optional model instance to be served.
        headless_mock: Whether to explicitly return an offline HeadlessSGLangRunner.
        device: Device to place the runner on.
        **kwargs: Extra arguments forwarded to runner initialization.

    Returns:
        HeadlessSGLangRunner when mock is active or SGLang is unavailable in mock mode.

    Raises:
        IntegrationNotImplementedError: When SGLang is unavailable and mock is not requested.
    """
    if not sglang_is_available():
        if headless_mock or os.environ.get("NEURALOPS_MOCK_SERVING") == "1":
            return HeadlessSGLangRunner(model=model, device=device, **kwargs)
        raise IntegrationNotImplementedError(
            "SGLang is not installed. Install with `pip install 'sglang>=0.4.0'` "
            "or set NEURALOPS_MOCK_SERVING=1 (or headless_mock=True) for offline mock runner."
        )

    # When SGLang is installed, construct native runtime or mock
    if headless_mock:
        return HeadlessSGLangRunner(model=model, device=device, **kwargs)

    try:
        from neuralops.integrations.sglang_backend.backend import NeuralOpsLinearAttnBackend

        return NeuralOpsLinearAttnBackend(model_runner=model, **kwargs)
    except Exception as exc:
        raise IntegrationNotImplementedError(f"Failed to initialize SGLang runtime runner: {exc}") from exc
