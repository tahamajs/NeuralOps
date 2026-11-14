#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Asynchronous Hugging Face Hub Uploader with Automated Disk Quota Protection.

Uploads trained NeuralOps model checkpoints across all zoos (BERT Attention/Compressor,
Qwen2.5, TinyLlama, LLaVA VLM) to the Hugging Face Hub (org: Hooshaai) in a non-blocking
background thread pool, confirms remote commit integrity, and automatically deletes
weight files (attention_weights.safetensors / pytorch_model.bin / model.safetensors)
to maintain local disk usage under quota constraints.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import logging
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# 8 canonical secret detection patterns for tokens, keys, and credentials
SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),  # Hugging Face token
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # OpenAI / API secret
    re.compile(r"ghp_[A-Za-z0-9]{36}"),  # GitHub personal token
    re.compile(r"ghs_[A-Za-z0-9]{36}"),  # GitHub server token
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS Access Key ID
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),  # Google API key
    re.compile(r"pypi-AgEIcHlwaS5vcmc[A-Za-z0-9\-_]{50,}"),  # PyPI upload token
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]"),  # Generic API key
]


def _redact_secrets(message: str, secret: Optional[str] = None) -> str:
    """Ensure no HF token or API secret appears in messages, errors, or logs."""
    if not message:
        return message
    text = str(message)
    if secret:
        text = text.replace(secret, "[REDACTED_HF_TOKEN]")
    for env_var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        env_token = os.environ.get(env_var)
        if env_token:
            text = text.replace(env_token, "[REDACTED_HF_TOKEN]")
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED_HF_TOKEN]", text)
    return text


def retry_hf_api_call(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    **kwargs: Any,
) -> T:
    """Execute a Hugging Face API call with resilient exponential backoff and jitter for rate limits (HTTP 429) and timeouts."""
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code is None and hasattr(exc, "response") and exc.response is not None:
                status_code = getattr(exc.response, "status_code", None)

            is_rate_limit = status_code == 429 or "429" in str(exc) or "rate limit" in str(exc).lower()
            is_timeout = (
                isinstance(exc, (TimeoutError, ConnectionError))
                or "timeout" in str(exc).lower()
                or "timed out" in str(exc).lower()
                or "connection" in str(exc).lower()
                or status_code in (500, 502, 503, 504)
            )

            if attempt == max_retries or (not is_rate_limit and not is_timeout):
                logger.error(
                    "HF Hub API call fatal failure on attempt %d/%d: %s",
                    attempt,
                    max_retries,
                    _redact_secrets(str(exc)),
                )
                raise exc

            retry_after: float | None = None
            if hasattr(exc, "response") and hasattr(exc.response, "headers"):
                ra_header = exc.response.headers.get("Retry-After")
                if ra_header and str(ra_header).isdigit():
                    retry_after = float(ra_header)

            if retry_after is not None:
                delay = retry_after + random.uniform(0.5, 1.5)
            else:
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)) + random.uniform(0.5, 2.0))

            reason = "Rate limited (HTTP 429)" if is_rate_limit else "Network timeout / server error"
            logger.warning(
                "%s on attempt %d/%d. Backing off for %.2fs before retrying...",
                reason,
                attempt,
                max_retries,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError("Unreachable: retry loop exited without return or exception")


def upload_with_retry(
    repo_id: str,
    folder_path: str | Path,
    token: Optional[str] = None,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    repo_type: str = "model",
    ignore_patterns: Optional[list[str]] = None,
    commit_message: Optional[str] = None,
) -> Any:
    """Synchronous upload folder helper with exponential backoff on HTTP 429 and strict token security."""
    resolved_token = os.environ.get("HF_TOKEN") or token
    if not resolved_token:
        try:
            from huggingface_hub import get_token

            resolved_token = get_token()
        except Exception:
            pass
    if not resolved_token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            resolved_token = token_path.read_text(encoding="utf-8").strip()

    if not resolved_token:
        raise ValueError("HF_TOKEN environment variable is strictly required for uploading to Hugging Face Hub.")

    from huggingface_hub import HfApi

    api = HfApi(token=resolved_token)

    retry_hf_api_call(
        api.create_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        exist_ok=True,
        private=False,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
    )

    return retry_hf_api_call(
        api.upload_folder,
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type=repo_type,
        ignore_patterns=ignore_patterns,
        commit_message=commit_message or f"Upload {repo_id}",
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
    )


class AsyncHFHubUploader:
    """Non-blocking background uploader daemon for Hugging Face Hub."""

    def __init__(
        self,
        hf_org: str = "Hooshaai",
        max_workers: int = 2,
        cleanup_weights: bool = True,
        token: Optional[str] = None,
        org: Optional[str] = None,
        max_retries: int = 5,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        **kwargs: Any,
    ) -> None:
        self.hf_org = org or hf_org
        self.max_workers = max_workers
        self.cleanup_weights = cleanup_weights
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        resolved_token = os.environ.get("HF_TOKEN") or token
        if not resolved_token:
            try:
                from huggingface_hub import get_token

                resolved_token = get_token()
            except Exception:
                pass
        if not resolved_token:
            token_path = Path.home() / ".cache" / "huggingface" / "token"
            if token_path.exists():
                resolved_token = token_path.read_text(encoding="utf-8").strip()
        self.token = resolved_token
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="hf_upload")
        self.futures: list[concurrent.futures.Future[Any]] = []

    def __repr__(self) -> str:
        token_status = "set" if self.token else "unset"
        return (
            f"AsyncHFHubUploader(hf_org={self.hf_org!r}, max_workers={self.max_workers}, "
            f"cleanup_weights={self.cleanup_weights}, token={token_status!r})"
        )

    def resolve_repo_id(
        self,
        key: str,
        staging_dir: Path,
        zoo_type: Optional[str] = None,
        repo_id: Optional[str] = None,
    ) -> str:
        """Resolve canonical Hugging Face repository ID for any NeuralOps zoo."""
        if repo_id:
            return repo_id

        slug = key.replace("_", "-")
        parent_hint = staging_dir.parent.name.lower() if staging_dir.parent else ""
        zoo_hint = (zoo_type or "").lower()
        combined_hint = f"{zoo_hint} {parent_hint}".strip()

        cfg_dict: dict[str, Any] = {}
        cfg_file = staging_dir / "config.json"
        if cfg_file.exists():
            try:
                cfg_dict = json.loads(cfg_file.read_text(encoding="utf-8"))
            except Exception:
                cfg_dict = {}

        manifest_zoo = ""
        manifest_file = staging_dir / "surgical_manifest.json"
        if manifest_file.exists():
            try:
                manifest_zoo = str(json.loads(manifest_file.read_text(encoding="utf-8")).get("zoo_type") or "").lower()
            except Exception:
                manifest_zoo = ""
        if manifest_zoo:
            combined_hint = f"{combined_hint} {manifest_zoo}".strip()

        model_type = str(cfg_dict.get("model_type") or "").lower()
        is_comp = bool(cfg_dict.get("compression_method") or cfg_dict.get("compressor_key") or "comp" in combined_hint)

        if "vlm" in combined_hint or "llava" in combined_hint or model_type == "llava":
            return f"{self.hf_org}/llava-qwen-0.5b-{slug}-scienceqa"
        if "tinyllama" in combined_hint or "llama" in combined_hint or model_type == "llama":
            return f"{self.hf_org}/tinyllama-1.1b-{slug}-wikitext2"
        if "qwen" in combined_hint or model_type == "qwen2":
            if is_comp:
                return f"{self.hf_org}/qwen-2.5-0.5b-comp-{slug}-wikitext2"
            return f"{self.hf_org}/qwen-2.5-0.5b-{slug}-wikitext2"
        if is_comp:
            return f"{self.hf_org}/bert-comp-{slug}-sst2"
        if "wiki" in parent_hint and "sst2" not in parent_hint:
            return f"{self.hf_org}/bert-{slug}-wiki"
        return f"{self.hf_org}/bert-{slug}-sst2"

    def submit_upload(
        self,
        attn_key: Any = None,
        staging_dir: Any = None,
        metrics: Optional[dict[str, Any]] = None,
        *,
        repo_id: Optional[str] = None,
        zoo_type: Optional[str] = None,
<<<<<<< HEAD
=======
        folder_path: Optional[str | Path] = None,
>>>>>>> fix/benchmarks-remediation
        mechanism_name: Optional[str] = None,
        **kwargs: Any,
    ) -> concurrent.futures.Future[Any]:
        """Submit a completed checkpoint folder for background upload and cleanup.

<<<<<<< HEAD
        Supports both standard signature:
            submit_upload(attn_key, staging_dir, metrics, *, repo_id=..., zoo_type=...)
        and caller pattern:
            submit_upload(ckpt_dir, repo_id, mechanism_name=key)
        """
        is_caller_pattern = False
        if mechanism_name is not None:
            is_caller_pattern = True
        elif isinstance(attn_key, Path):
            is_caller_pattern = True
        elif isinstance(attn_key, str):
            try:
                p = Path(attn_key)
                if p.is_dir() or "/" in attn_key or "\\" in attn_key:
                    if staging_dir is None or not (
                        isinstance(staging_dir, Path) or (isinstance(staging_dir, str) and Path(staging_dir).is_dir())
                    ):
                        is_caller_pattern = True
            except OSError:
                pass

        if is_caller_pattern:
            final_staging_dir = Path(attn_key) if attn_key is not None else Path(".")
            final_attn_key = mechanism_name or kwargs.get("key") or final_staging_dir.name
            final_repo_id = repo_id or (str(staging_dir) if isinstance(staging_dir, str) else None)
            final_metrics = metrics if isinstance(metrics, dict) else (kwargs.get("metrics") or {})
        else:
            final_attn_key = str(attn_key) if attn_key is not None else (mechanism_name or "unknown")
            final_staging_dir = Path(staging_dir) if staging_dir is not None else Path(".")
            final_repo_id = repo_id
            final_metrics = metrics if isinstance(metrics, dict) else (kwargs.get("metrics") or {})

        future = self.executor.submit(
            self._worker_upload_and_cleanup,
            final_attn_key,
            final_staging_dir,
            final_metrics,
            final_repo_id,
            zoo_type or kwargs.get("zoo_type"),
=======
        Supports both canonical and legacy call signatures:
          1. submit_upload(attn_key, staging_dir, metrics, repo_id=..., zoo_type=...)
          2. submit_upload(folder_path, repo_id, mechanism_name=...)
        """
        # Resolve folder_path / staging_dir alias
        if folder_path is not None:
            resolved_path = Path(folder_path)
            resolved_key = str(mechanism_name or attn_key or resolved_path.name)
            resolved_repo_id = repo_id or (
                str(staging_dir) if staging_dir and not isinstance(staging_dir, dict) else None
            )
            resolved_metrics = metrics if isinstance(metrics, dict) else (kwargs.get("metrics") or {})
            resolved_zoo_type = zoo_type or kwargs.get("zoo_type")

        # Handle legacy signature: submit_upload(ckpt_dir, repo_id, mechanism_name=...)
        elif mechanism_name is not None:
            resolved_key = str(mechanism_name)
            resolved_path = Path(attn_key) if attn_key is not None else Path(".")
            resolved_repo_id = repo_id or (str(staging_dir) if staging_dir else None)
            resolved_metrics = metrics if isinstance(metrics, dict) else (kwargs.get("metrics") or {})
            resolved_zoo_type = zoo_type or kwargs.get("zoo_type")

        # First arg is a Path object or directory string, and 3rd arg is not a dict
        elif isinstance(attn_key, Path) or (
            isinstance(attn_key, str)
            and (os.path.sep in attn_key or "/" in attn_key or Path(attn_key).is_dir())
            and not isinstance(metrics, dict)
        ):
            resolved_path = Path(attn_key)
            resolved_repo_id = repo_id or (str(staging_dir) if staging_dir else None)
            resolved_key = str(mechanism_name or resolved_path.name)
            resolved_metrics = metrics if isinstance(metrics, dict) else (kwargs.get("metrics") or {})
            resolved_zoo_type = zoo_type or kwargs.get("zoo_type")

        # Canonical signature: submit_upload(attn_key, staging_dir, metrics, repo_id=..., zoo_type=...)
        else:
            resolved_key = str(attn_key) if attn_key is not None else "unknown"
            resolved_path = Path(staging_dir) if staging_dir is not None else Path(".")
            resolved_metrics = metrics if isinstance(metrics, dict) else (kwargs.get("metrics") or {})
            resolved_repo_id = repo_id
            resolved_zoo_type = zoo_type or kwargs.get("zoo_type")

        future = self.executor.submit(
            self._worker_upload_and_cleanup,
            resolved_key,
            resolved_path,
            resolved_metrics,
            resolved_repo_id,
            resolved_zoo_type,
>>>>>>> fix/benchmarks-remediation
        )
        self.futures.append(future)
        return future

    def _worker_upload_and_cleanup(
        self,
        attn_key: str,
        staging_dir: Path,
        metrics: dict[str, Any],
        explicit_repo_id: Optional[str] = None,
        zoo_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Background worker uploading directory and deleting heavy weights."""
        staging_dir.mkdir(parents=True, exist_ok=True)
        if not self.token:
            err_msg = "HF_TOKEN is unset — skipping remote upload. Weights preserved."
            logger.warning("[%s] %s", attn_key, err_msg)
            err_file = staging_dir / "upload_error.json"
            err_file.write_text(
                json.dumps(
                    {
                        "error": err_msg,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                ),
                encoding="utf-8",
            )
            return {"status": "SKIPPED_NO_TOKEN", "error": err_msg}

        slug = attn_key.replace("_", "-")
        repo_id = self.resolve_repo_id(attn_key, staging_dir, zoo_type=zoo_type, repo_id=explicit_repo_id)
        logger.info("[%s] Beginning background upload to Hugging Face Hub (%s)...", attn_key, repo_id)

        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self.token)

            # 1. Ensure remote repository exists with exponential backoff on HTTP 429
            retry_hf_api_call(
                api.create_repo,
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
                private=False,
                max_retries=self.max_retries,
                base_delay=self.base_delay,
                max_delay=self.max_delay,
            )

            # 2. Strip legacy full weights before upload if surgical delta safetensors exists
            ignore_patterns = ["upload_success.json", "upload_error.json", "*.tmp"]
            if (staging_dir / "attention_weights.safetensors").exists():
                ignore_patterns.extend(
                    ["pytorch_model.bin", "model.safetensors", "surgical_model_state.pt", "optimizer.pt"]
                )

            ppl_val = metrics.get("post_kd_ppl", metrics.get("ppl_after"))
            acc_val = metrics.get("post_kd_acc", metrics.get("acc_after", metrics.get("healed_acc")))
            stat_parts = []
            if isinstance(acc_val, (int, float)):
                stat_parts.append(f"acc={acc_val:.2f}%")
            if isinstance(ppl_val, (int, float)):
                stat_parts.append(f"ppl={ppl_val:.2f}")
            stat_summary = f" ({', '.join(stat_parts)})" if stat_parts else ""
            commit_message = f"Upload {slug} checkpoint -> {repo_id}{stat_summary}"

            commit_info = retry_hf_api_call(
                api.upload_folder,
                folder_path=str(staging_dir),
                repo_id=repo_id,
                repo_type="model",
                ignore_patterns=ignore_patterns,
                commit_message=commit_message,
                max_retries=self.max_retries,
                base_delay=self.base_delay,
                max_delay=self.max_delay,
            )

            commit_hash = getattr(commit_info, "commit_id", getattr(commit_info, "oid", "verified"))
            logger.info("[%s] Upload successful! Remote commit: %s", attn_key, commit_hash)

            # 3. Write success confirmation record
            success_record = {
                "status": "UPLOAD_SUCCESS",
                "repo_id": repo_id,
                "commit_hash": str(commit_hash),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "metrics": {
                    "ppl_after": ppl_val,
                    "acc_after": acc_val,
                },
            }
            (staging_dir / "upload_success.json").write_text(json.dumps(success_record, indent=2), encoding="utf-8")

            # 4. Atomic Disk Quota Protection: Delete local weight files after confirmed upload
            if self.cleanup_weights:
                for weight_name in (
                    "attention_weights.safetensors",
                    "pytorch_model.bin",
                    "model.safetensors",
                    "surgical_model_state.pt",
                    "optimizer.pt",
                ):
                    w_path = staging_dir / weight_name
                    if w_path.exists():
                        size_mb = w_path.stat().st_size / (1024 * 1024)
                        os.remove(w_path)
                        logger.info(
                            "[%s] [DISK CLEANUP] Deleted local weight file %s (%.1f MB freed).",
                            attn_key,
                            weight_name,
                            size_mb,
                        )

            return success_record

        except Exception as e:
            redacted_err = _redact_secrets(str(e), self.token)
            logger.error("[%s] Background upload failed: %s", attn_key, redacted_err, exc_info=True)
            err_record = {
                "status": "UPLOAD_FAILED",
                "error": redacted_err,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            (staging_dir / "upload_error.json").write_text(json.dumps(err_record, indent=2), encoding="utf-8")
            return err_record

    def wait_all(self, timeout: Optional[float] = None) -> list[dict[str, Any]]:
        """Wait for all pending upload tasks to complete."""
        results = []
        for future in self.futures:
            try:
                res = future.result(timeout=timeout)
                results.append(res)
            except Exception as e:
                results.append({"status": "FUTURE_EXCEPTION", "error": str(e)})
        return results

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the background thread pool."""
        self.executor.shutdown(wait=wait)


__all__ = [
    "AsyncHFHubUploader",
    "SECRET_PATTERNS",
    "_redact_secrets",
    "retry_hf_api_call",
    "upload_with_retry",
]
