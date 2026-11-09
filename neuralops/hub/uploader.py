#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Asynchronous Hugging Face Hub Uploader with Automated Disk Quota Protection.

Uploads trained BERT attention model checkpoints to the Hugging Face Hub in
a non-blocking background thread pool, confirms remote commit integrity, and
automatically deletes heavy local weight files (.safetensors / pytorch_model.bin)
to maintain local disk usage under quota constraints.
"""
from __future__ import annotations

import concurrent.futures
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AsyncHFHubUploader:
    """
    Non-blocking background uploader daemon for Hugging Face Hub.
    """

    def __init__(
        self,
        hf_org: str = "Hooshaai",
        max_workers: int = 2,
        cleanup_weights: bool = True,
        token: Optional[str] = None,
    ) -> None:
        self.hf_org = hf_org
        self.max_workers = max_workers
        self.cleanup_weights = cleanup_weights
        self.token = token or os.environ.get("HF_TOKEN")
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="hf_upload"
        )
        self.futures: list[concurrent.futures.Future[Any]] = []

    def submit_upload(
        self,
        attn_key: str,
        staging_dir: str | Path,
        metrics: dict[str, Any],
    ) -> concurrent.futures.Future[Any]:
        """Submit a completed checkpoint folder for background upload and cleanup."""
        path = Path(staging_dir)
        future = self.executor.submit(
            self._worker_upload_and_cleanup,
            attn_key,
            path,
            metrics,
        )
        self.futures.append(future)
        return future

    def _worker_upload_and_cleanup(
        self,
        attn_key: str,
        staging_dir: Path,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Background worker uploading directory and deleting heavy weights."""
        if not self.token:
            err_msg = "HF_TOKEN is unset — skipping remote upload. Weights preserved."
            logger.warning("[%s] %s", attn_key, err_msg)
            err_file = staging_dir / "upload_error.json"
            err_file.write_text(json.dumps({"error": err_msg, "timestamp": datetime.datetime.utcnow().isoformat()}))
            return {"status": "SKIPPED_NO_TOKEN", "error": err_msg}

        slug = attn_key.replace("_", "-")
        repo_id = f"{self.hf_org}/bert-{slug}-wiki"
        logger.info("[%s] Beginning background upload to Hugging Face Hub (%s)...", attn_key, repo_id)

        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self.token)

            # 1. Ensure remote repository exists
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

            # 2. Upload entire folder to Hugging Face Hub
            commit_message = (
                f"Distill bert-{slug} on WikiText-2 "
                f"(PPL_after={metrics.get('post_kd_ppl', metrics.get('ppl_after', 0.0)):.2f})"
            )
            commit_info = api.upload_folder(
                folder_path=str(staging_dir),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )

            commit_hash = getattr(commit_info, "commit_id", "verified")
            logger.info("[%s] Upload successful! Remote commit: %s", attn_key, commit_hash)

            # 3. Write success confirmation record
            success_record = {
                "status": "UPLOAD_SUCCESS",
                "repo_id": repo_id,
                "commit_hash": str(commit_hash),
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "metrics": {
                    "ppl_after": metrics.get("post_kd_ppl", metrics.get("ppl_after", 0.0)),
                    "acc_after": metrics.get("post_kd_acc", metrics.get("acc_after", 0.0)),
                },
            }
            (staging_dir / "upload_success.json").write_text(json.dumps(success_record, indent=2))

            # 4. Atomic Disk Quota Protection: Delete heavy model weights
            if self.cleanup_weights:
                for weight_name in ["model.safetensors", "pytorch_model.bin", "optimizer.pt"]:
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
            logger.error("[%s] Background upload failed: %s", attn_key, e, exc_info=True)
            err_record = {
                "status": "UPLOAD_FAILED",
                "error": str(e),
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
            (staging_dir / "upload_error.json").write_text(json.dumps(err_record, indent=2))
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
