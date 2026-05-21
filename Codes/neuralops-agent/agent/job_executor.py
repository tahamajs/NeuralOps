"""
Job lifecycle manager — handles the full lifecycle of a compute job.

Sequence:
  1. Receive job from WebSocket
  2. Pull Docker image
  3. Run container
  4. Monitor container (logs, status)
  5. On completion: upload results, verify, report
  6. On failure: capture logs, report
  7. Cleanup
"""
import logging, threading, time, os, hashlib
from pathlib import Path

logger = logging.getLogger("neuralops.executor")

class JobExecutor:
    def __init__(self, docker_manager, api_client):
        self.docker = docker_manager
        self.api = api_client
        self._active_jobs: dict = {}  # job_id -> container_id
        self._lock = threading.Lock()

    def execute(self, job_data: dict):
        """Execute a compute job in background thread."""
        job_id = job_data['id']
        with self._lock:
            if job_id in self._active_jobs:
                logger.warning("Job %s already running", job_id)
                return
            self._active_jobs[job_id] = None

        thread = threading.Thread(
            target=self._run_job, args=(job_data,), daemon=True
        )
        thread.start()

    def cancel(self, job_id: str):
        """Cancel a running job."""
        with self._lock:
            container_id = self._active_jobs.pop(job_id, None)
        if container_id:
            self.docker.stop_container(container_id)
            self.api.report_job_failed(job_id, 'Cancelled by user')
            logger.info("Job %s cancelled", job_id)

    def _run_job(self, job_data: dict):
        """Run a single compute job to completion."""
        job_id = job_data['id']
        image = job_data.get('docker_image', 'pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime')
        command = job_data.get('command', '')
        input_url = job_data.get('input_data_url', '')
        env_vars = job_data.get('env_vars', {})

        try:
            # 1. Pull image
            self.api.report_job_status(job_id, 'provisioning')
            self.docker.pull_image(image)

            # 2. Run container
            self.api.report_job_status(job_id, 'running')
            container_id = self.docker.run_job(
                job_id=job_id, image=image, command=command,
                env_vars=env_vars, input_data_url=input_url
            )
            with self._lock:
                self._active_jobs[job_id] = container_id

            # 3. Monitor until completion
            self._monitor(job_id, container_id)

        except Exception as e:
            logger.exception("Job %s failed: %s", job_id, e)
            self.api.report_job_failed(job_id, str(e))
        finally:
            with self._lock:
                self._active_jobs.pop(job_id, None)

    def _monitor(self, job_id: str, container_id: str):
        """Monitor a container until it exits."""
        last_log_tail = 0
        while True:
            try:
                status = self.docker.get_container_status(container_id)
                if status == 'exited':
                    # Get full logs
                    logs = self.docker.get_container_logs(container_id, tail=0)
                    exit_code = self._get_exit_code(container_id)

                    if exit_code == 0:
                        # Success: compute hash of outputs for verification
                        output_hash = self._compute_output_hash(job_id)
                        self.api.report_job_completed(
                            job_id, output_hash=output_hash, logs=logs
                        )
                        logger.info("Job %s completed successfully", job_id)
                    else:
                        self.api.report_job_failed(
                            job_id, f"Container exited with code {exit_code}", logs=logs
                        )
                        logger.warning("Job %s failed with exit code %d", job_id, exit_code)
                    break

                elif status == 'not_found':
                    self.api.report_job_failed(job_id, 'Container disappeared')
                    break

                # Stream recent logs
                logs = self.docker.get_container_logs(container_id, tail=50)
                if logs and len(logs) > last_log_tail:
                    self.api.report_job_logs(job_id, logs[last_log_tail:])
                    last_log_tail = len(logs)

                time.sleep(5)

            except Exception as e:
                logger.error("Monitor error for job %s: %s", job_id, e)
                time.sleep(10)

    def _get_exit_code(self, container_id: str) -> int:
        try:
            container = self.docker.client.containers.get(container_id)
            return container.attrs['State']['ExitCode']
        except Exception:
            return -1

    def _compute_output_hash(self, job_id: str) -> str:
        """Compute SHA-256 hash of job output directory for verification."""
        job_dir = f'/tmp/neuralops-jobs/{job_id}'
        sha = hashlib.sha256()
        for root, _, files in os.walk(job_dir):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                sha.update(fname.encode())
                with open(fpath, 'rb') as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        sha.update(chunk)
        return sha.hexdigest()