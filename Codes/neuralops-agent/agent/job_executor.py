"""
Job Lifecycle Manager — robust execution, state transitions, timeout watchdogs,
log streaming, artifact verification, and resource cleanup.
"""

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("neuralops.executor")


class JobState(str, Enum):
    QUEUED = "queued"
    DOWNLOADING_DATA = "downloading_data"
    RUNNING = "running"
    SAVING_ARTIFACTS = "saving_artifacts"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobContext:
    """Metadata and execution context for a single active compute job."""

    def __init__(self, job_id: str, job_data: dict, job_dir: str):
        self.job_id: str = job_id
        self.job_data: dict = job_data
        self.job_dir: str = job_dir
        self.state: JobState = JobState.QUEUED
        self.container_id: Optional[str] = None
        self.start_time: float = time.time()
        self.timeout_seconds: int = int(
            job_data.get("timeout_seconds")
            or job_data.get("max_execution_time")
            or 3600
        )
        self.cancel_event: threading.Event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.exit_code: Optional[int] = None
        self.error: Optional[str] = None
        self.output_hash: str = ""
        self.artifacts_manifest: List[Dict[str, Any]] = []
        self.log_buffer: List[str] = []


class JobExecutor:
    """
    Executes compute jobs across their full lifecycle with strict state tracking,
    timeout watchdogs, streaming logs, and guaranteed cleanup.
    """

    def __init__(
        self,
        docker_manager,
        api_client,
        ws_client=None,
        max_concurrent_jobs: int = 2,
        base_job_dir: str = "/tmp/neuralops-jobs",
    ):
        self.docker = docker_manager
        self.api = api_client
        self.ws = ws_client
        self.max_concurrent_jobs = max_concurrent_jobs
        self.base_job_dir = base_job_dir
        self._active_jobs: Dict[str, JobContext] = {}
        self._lock = threading.Lock()

    def set_websocket_client(self, ws_client):
        """Set or update WebSocket client for real-time streaming."""
        self.ws = ws_client

    def execute(self, job_data: dict) -> bool:
        """
        Accept and execute a compute job in a dedicated background thread.
        Transitions job through QUEUED -> DOWNLOADING_DATA -> RUNNING -> SAVING_ARTIFACTS -> COMPLETED/FAILED.
        """
        job_id = job_data.get("id")
        if not job_id:
            logger.error("Rejecting job execution: missing job id")
            return False

        with self._lock:
            if job_id in self._active_jobs:
                logger.warning("Job %s is already tracked as active", job_id)
                return False

            # Check concurrency limit
            running_count = sum(
                1 for ctx in self._active_jobs.values()
                if ctx.state not in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED)
            )
            if running_count >= self.max_concurrent_jobs:
                logger.warning(
                    "Concurrency limit reached (%d/%d). Rejecting job %s",
                    running_count,
                    self.max_concurrent_jobs,
                    job_id,
                )
                self._report_failure_direct(
                    job_id,
                    f"Agent concurrency limit reached ({running_count}/{self.max_concurrent_jobs})",
                )
                return False

            job_dir = os.path.join(self.base_job_dir, job_id)
            context = JobContext(job_id=job_id, job_data=job_data, job_dir=job_dir)
            self._active_jobs[job_id] = context

        # Report initial Queued state
        self._transition_state(context, JobState.QUEUED, "Job accepted into execution queue")

        thread = threading.Thread(
            target=self._run_job_pipeline,
            args=(context,),
            name=f"job-runner-{job_id[:8]}",
            daemon=True,
        )
        context.thread = thread
        thread.start()
        return True

    def cancel(self, job_id: str, reason: str = "Cancelled by user") -> bool:
        """Cancel a running job immediately, kill its container, and clean up."""
        context = None
        with self._lock:
            context = self._active_jobs.get(job_id)

        if not context:
            logger.warning("Attempted to cancel unknown or inactive job %s", job_id)
            return False

        logger.info("Cancelling job %s: %s", job_id, reason)
        context.cancel_event.set()
        context.error = reason

        # Terminate container immediately
        if context.container_id:
            try:
                self.docker.stop_container(context.container_id, timeout=2, remove=True)
            except Exception as e:
                logger.warning("Error stopping container on cancel for job %s: %s", job_id, e)

        self._transition_state(context, JobState.CANCELLED, reason)
        self.api.report_job_failed(job_id, f"Job cancelled: {reason}")
        if self.ws:
            self.ws.send_job_update(job_id, "cancelled", {"reason": reason})

        with self._lock:
            self._active_jobs.pop(job_id, None)

        return True

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve current execution status and metadata for a job."""
        with self._lock:
            ctx = self._active_jobs.get(job_id)
            if not ctx:
                return None
            return {
                "job_id": ctx.job_id,
                "state": ctx.state.value,
                "container_id": ctx.container_id,
                "start_time": ctx.start_time,
                "elapsed_seconds": round(time.time() - ctx.start_time, 1),
                "timeout_seconds": ctx.timeout_seconds,
                "exit_code": ctx.exit_code,
                "error": ctx.error,
                "output_hash": ctx.output_hash,
            }

    def get_active_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve snapshot of all currently active jobs."""
        with self._lock:
            return {
                jid: {
                    "state": ctx.state.value,
                    "container_id": ctx.container_id,
                    "elapsed_seconds": round(time.time() - ctx.start_time, 1),
                }
                for jid, ctx in self._active_jobs.items()
            }

    # ── Internal Execution Pipeline ───────────────────────────────

    def _run_job_pipeline(self, ctx: JobContext):
        """Execute all phases of the job lifecycle."""
        job_id = ctx.job_id
        job_data = ctx.job_data
        image = job_data.get(
            "docker_image",
            job_data.get("image", "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"),
        )
        command = job_data.get("command", "")
        input_url = job_data.get("input_data_url", "")
        model_id = job_data.get("model_id", "")
        env_vars = job_data.get("env_vars", {})
        gpu_devices = job_data.get("gpu_devices", "all")
        gpu_count = job_data.get("gpu_count")
        use_gpu = job_data.get("use_gpu", True)
        cpu_limit = job_data.get("cpu_limit")
        memory_limit = job_data.get("memory_limit", "8g")
        shm_size = job_data.get("shm_size", "2g")
        preserve_workspace = job_data.get("preserve_workspace", False)

        try:
            # ── Phase 1: Downloading Data & Image Provisioning ─────
            self._transition_state(
                ctx,
                JobState.DOWNLOADING_DATA,
                f"Pulling container image {image} and preparing workspace",
            )

            if ctx.cancel_event.is_set():
                raise RuntimeError("Job cancelled prior to image pull")

            # Pull image
            self.docker.pull_image(image)

            # Build isolation workspace
            workspace_dir = os.path.join(ctx.job_dir, "workspace")
            data_dir = os.path.join(workspace_dir, "data")
            artifacts_dir = os.path.join(workspace_dir, "artifacts")
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(artifacts_dir, exist_ok=True)

            # Build entrypoint script (handling data/model downloads)
            pre_script = self._build_pre_script(
                model_id=model_id,
                input_url=input_url,
                custom_script=job_data.get("pre_script"),
            )

            if ctx.cancel_event.is_set():
                raise RuntimeError("Job cancelled during data preparation")

            # ── Phase 2: Container Launch & Running ───────────────
            self._transition_state(
                ctx,
                JobState.RUNNING,
                "Container initialized, beginning workload execution",
            )

            container_id = self.docker.run_job_with_pre_script(
                job_id=job_id,
                image=image,
                command=command,
                env_vars=env_vars,
                pre_script=pre_script,
                input_data_url=input_url,
                gpu_devices=gpu_devices,
                gpu_count=gpu_count,
                use_gpu=use_gpu,
                cpu_limit=cpu_limit,
                memory_limit=memory_limit,
                shm_size=shm_size,
                job_dir=ctx.job_dir,
                read_only=job_data.get("read_only", True),
            )
            ctx.container_id = container_id

            # ── Phase 3: Monitor & Stream Logs with Timeout Watchdog
            exit_code, container_info = self._monitor_and_stream(ctx)
            ctx.exit_code = exit_code

            if ctx.cancel_event.is_set():
                raise RuntimeError("Job cancelled during execution")

            if container_info.get("oom_killed"):
                raise RuntimeError(
                    f"Job exceeded memory limit ({memory_limit}) and was killed by Linux OOM killer"
                )

            if exit_code != 0:
                raise RuntimeError(
                    f"Container process exited with non-zero exit code: {exit_code}"
                )

            # ── Phase 4: Saving & Verifying Artifacts ──────────────
            self._transition_state(
                ctx,
                JobState.SAVING_ARTIFACTS,
                "Collecting artifacts and calculating integrity checksums",
            )

            output_hash, manifest = self._compute_output_artifacts(ctx.job_dir)
            ctx.output_hash = output_hash
            ctx.artifacts_manifest = manifest

            # ── Phase 5: Completed ────────────────────────────────
            self._transition_state(
                ctx,
                JobState.COMPLETED,
                f"Job completed successfully. Output Hash: {output_hash[:12]}",
            )

            final_logs = self.docker.get_container_logs(container_id, tail=1000)
            self.api.report_job_completed(
                job_id=job_id,
                output_hash=output_hash,
                logs=final_logs,
            )

            if self.ws:
                self.ws.send_job_update(
                    job_id=job_id,
                    status="completed",
                    data={
                        "output_hash": output_hash,
                        "manifest": manifest,
                        "duration_seconds": round(time.time() - ctx.start_time, 2),
                    },
                )
            logger.info("Job %s completed successfully (duration: %.1fs)", job_id, time.time() - ctx.start_time)

        except Exception as e:
            error_msg = str(e)
            ctx.error = error_msg
            logger.exception("Job %s failed: %s", job_id, error_msg)

            self._transition_state(ctx, JobState.FAILED, error_msg)

            logs = ""
            if ctx.container_id:
                try:
                    logs = self.docker.get_container_logs(ctx.container_id, tail=500)
                except Exception:
                    pass

            self.api.report_job_failed(job_id, error_msg, logs=logs)
            if self.ws:
                self.ws.send_job_update(
                    job_id=job_id,
                    status="failed",
                    data={"error": error_msg, "logs": logs[-2000:] if logs else ""},
                )

        finally:
            # ── Phase 6: Cleanup ──────────────────────────────────
            self._cleanup_job(ctx, preserve_workspace=preserve_workspace)

    def _monitor_and_stream(self, ctx: JobContext) -> Tuple[int, Dict[str, Any]]:
        """
        Monitor container execution, stream logs incrementally, and enforce timeout watchdog.
        """
        container_id = ctx.container_id
        job_id = ctx.job_id
        start_time = ctx.start_time
        timeout = ctx.timeout_seconds
        last_log_offset = 0

        # Dedicated log streaming thread to capture continuous stdout/stderr
        log_thread_stop = threading.Event()

        def stream_worker():
            try:
                for chunk in self.docker.stream_logs(container_id, follow=True, tail=0):
                    if log_thread_stop.is_set():
                        break
                    if chunk:
                        ctx.log_buffer.append(chunk)
                        # Stream incrementally to API & WebSocket
                        try:
                            self.api.report_job_logs(job_id, chunk)
                            if self.ws:
                                self.ws.send_container_logs(job_id, chunk)
                        except Exception:
                            pass
            except Exception as stream_err:
                logger.debug("Log stream worker ended for %s: %s", job_id, stream_err)

        log_thread = threading.Thread(
            target=stream_worker, name=f"log-stream-{job_id[:8]}", daemon=True
        )
        log_thread.start()

        try:
            while True:
                # 1. Check cancellation
                if ctx.cancel_event.is_set():
                    raise RuntimeError("Job cancelled by user")

                # 2. Check timeout watchdog
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    logger.warning(
                        "Timeout watchdog triggered for job %s (elapsed %.1fs > %ds)",
                        job_id,
                        elapsed,
                        timeout,
                    )
                    self.docker.stop_container(container_id, timeout=2, remove=True)
                    raise TimeoutError(f"Job execution timed out after {timeout} seconds")

                # 3. Check container status
                info = self.docker.get_container_info(container_id)
                status = info.get("status")

                if status == "exited":
                    exit_code = info.get("exit_code", -1)
                    return exit_code, info
                elif status in ("not_found", "dead"):
                    raise RuntimeError(f"Container terminated unexpectedly with status '{status}'")

                time.sleep(1.0)

        finally:
            log_thread_stop.set()
            if log_thread.is_alive():
                log_thread.join(timeout=1.0)

    def _build_pre_script(
        self,
        model_id: Optional[str] = None,
        input_url: Optional[str] = None,
        custom_script: Optional[str] = None,
    ) -> str:
        """
        Generate a bash entrypoint script that downloads models and data into /workspace.
        """
        script_lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            "export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH",
            "cd /workspace",
            'echo "=== [NeuralOps] Initializing Workspace at $(date) ==="',
        ]

        if custom_script and custom_script.strip():
            script_lines.append("\n# Custom Setup Script")
            script_lines.append(custom_script.strip())

        if model_id and model_id.strip():
            script_lines.append(f'\necho "=== [NeuralOps] Downloading Model: {model_id} ==="')
            script_lines.append(
                f'python3 -c "'
                f'import sys\n'
                f'try:\n'
                f'    from transformers import AutoModel, AutoTokenizer\n'
                f'    AutoTokenizer.from_pretrained(\'{model_id}\')\n'
                f'    AutoModel.from_pretrained(\'{model_id}\')\n'
                f'    print(\'Model downloaded successfully.\')\n'
                f'except Exception as e:\n'
                f'    print(f\'HuggingFace download notice: {{e}}\', file=sys.stderr)\n'
                f'" || true'
            )

        if input_url and input_url.strip():
            script_lines.append(f'\necho "=== [NeuralOps] Downloading Dataset from {input_url} ==="')
            script_lines.append('mkdir -p /workspace/data')
            script_lines.append(
                f'if command -v curl &>/dev/null; then\n'
                f'    curl -fsSL -o /workspace/data/input_data.tar.gz "{input_url}"\n'
                f'elif command -v wget &>/dev/null; then\n'
                f'    wget -q -O /workspace/data/input_data.tar.gz "{input_url}"\n'
                f'fi'
            )
            script_lines.append(
                'if [ -f /workspace/data/input_data.tar.gz ]; then\n'
                '    tar -xzf /workspace/data/input_data.tar.gz -C /workspace/data/ || true\n'
                'fi'
            )

        script_lines.append('\necho "=== [NeuralOps] Starting Primary Workload ==="')
        return "\n".join(script_lines)

    def _compute_output_artifacts(self, job_dir: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Compute SHA256 hashes of generated artifacts and generate an integrity manifest.
        """
        workspace_dir = os.path.join(job_dir, "workspace")
        artifacts_dir = os.path.join(workspace_dir, "artifacts")
        target_dir = artifacts_dir if os.path.exists(artifacts_dir) else workspace_dir

        sha_composite = hashlib.sha256()
        manifest: List[Dict[str, Any]] = []

        for root, _, files in os.walk(target_dir):
            for fname in sorted(files):
                # Skip entrypoint scripts and internal logs
                if fname in ("entrypoint.sh", ".DS_Store"):
                    continue

                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, workspace_dir)
                try:
                    fsize = os.path.getsize(fpath)
                    fhash = hashlib.sha256()
                    with open(fpath, "rb") as f:
                        while chunk := f.read(65536):
                            fhash.update(chunk)

                    file_sha = fhash.hexdigest()
                    sha_composite.update(rel_path.encode("utf-8"))
                    sha_composite.update(file_sha.encode("utf-8"))

                    manifest.append({
                        "path": rel_path,
                        "size_bytes": fsize,
                        "sha256": file_sha,
                    })
                except Exception as e:
                    logger.warning("Error hashing artifact %s: %s", fpath, e)

        composite_digest = sha_composite.hexdigest()
        return composite_digest, manifest

    def _transition_state(self, ctx: JobContext, new_state: JobState, description: str = ""):
        """Update job state and notify API / WebSocket."""
        ctx.state = new_state
        logger.info("[Job %s] State -> %s: %s", ctx.job_id[:8], new_state.value, description)

        try:
            self.api.report_job_status(ctx.job_id, new_state.value, description=description)
        except Exception as e:
            logger.debug("Failed reporting status to API for job %s: %s", ctx.job_id, e)

        if self.ws:
            try:
                self.ws.send_job_update(
                    ctx.job_id,
                    new_state.value,
                    {"description": description, "timestamp": time.time()},
                )
            except Exception:
                pass

    def _cleanup_job(self, ctx: JobContext, preserve_workspace: bool = False):
        """Clean up Docker container, remove active tracker, and optionally clean disk."""
        job_id = ctx.job_id

        # 1. Stop and remove container
        if ctx.container_id:
            try:
                self.docker.stop_container(ctx.container_id, timeout=5, remove=True)
            except Exception as e:
                logger.debug("Container cleanup note for %s: %s", ctx.container_id, e)

        # 2. Clean temporary files if not preserved
        if not preserve_workspace and os.path.exists(ctx.job_dir):
            try:
                # Remove entrypoint and temporary data downloads
                entrypoint = os.path.join(ctx.job_dir, "workspace", "entrypoint.sh")
                if os.path.exists(entrypoint):
                    os.remove(entrypoint)
            except Exception:
                pass

        # 3. Unregister from active jobs
        with self._lock:
            self._active_jobs.pop(job_id, None)

    def _report_failure_direct(self, job_id: str, error: str):
        """Helper to report failure when job context could not be created."""
        try:
            self.api.report_job_failed(job_id, error)
            if self.ws:
                self.ws.send_job_update(job_id, "failed", {"error": error})
        except Exception:
            pass
