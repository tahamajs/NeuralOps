"""
Docker container runtime manager.

Key security principles:
  - All user code runs inside isolated Docker containers
  - GPU access via NVIDIA Container Toolkit / device pass-through
  - Resource limits: memory, CPU, SHM, PIDs limits
  - Read-only root filesystem with tmpfs scratch space
  - Isolated volume mounts for workspace, data, and artifacts
  - Real-time streaming log capture
  - Auto-removal and graceful cleanup on completion/error

Inspired by: isol8 sandbox, NVIDIA DGX Spark secure containers,
and Boxed's sovereign code execution engine.
"""

import logging
import os
import shutil
import time
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import docker
from docker.errors import APIError, DockerException, NotFound
from docker.types import DeviceRequest

logger = logging.getLogger("neuralops.docker")


class DockerManager:
    """Manage Docker containers for executing compute jobs."""

    def __init__(
        self,
        default_shm_size: str = "2g",
        default_mem_limit: str = "8g",
        default_cpus: float = 4.0,
        default_pids_limit: int = 2048,
    ):
        self._client: Optional[docker.DockerClient] = None
        self.has_gpu_runtime: bool = False
        self.default_shm_size = default_shm_size
        self.default_mem_limit = default_mem_limit
        self.default_cpus = default_cpus
        self.default_pids_limit = default_pids_limit

    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def verify_runtime(self, test_image: Optional[str] = None) -> bool:
        """
        Verify Docker daemon and NVIDIA Container Toolkit are operational.
        Falls back to CPU container execution if NVIDIA runtime is unavailable.
        """
        try:
            # 1. Ping Docker daemon
            self.client.ping()
            logger.info("Docker daemon is running and reachable")

            # 2. Check NVIDIA GPU runtime
            cuda_image = test_image or "nvidia/cuda:12.4.0-base-ubuntu22.04"
            try:
                device_reqs = [
                    DeviceRequest(count=-1, capabilities=[["gpu"]])
                ]
                self.client.containers.run(
                    cuda_image,
                    command="nvidia-smi",
                    device_requests=device_reqs,
                    remove=True,
                    stdout=True,
                    stderr=True,
                )
                logger.info("NVIDIA container runtime verified with GPU pass-through")
                self.has_gpu_runtime = True
                return True
            except Exception as e:
                logger.warning(
                    "NVIDIA container runtime not configured or GPU unavailable: %s. "
                    "Testing CPU container fallback...",
                    e,
                )

            # 3. Fallback: verify basic CPU container run works
            try:
                self.client.containers.run(
                    "alpine:latest",
                    command='echo "NeuralOps standard CPU runtime test"',
                    remove=True,
                    stdout=True,
                    stderr=True,
                )
                logger.info("Standard CPU container runtime verified")
                self.has_gpu_runtime = False
                return True
            except Exception as cpu_err:
                logger.error("Standard CPU container runtime test failed: %s", cpu_err)
                self.has_gpu_runtime = False
                return False

        except DockerException as e:
            logger.critical("Docker daemon is not running: %s", e)
            self.has_gpu_runtime = False
            return False
        except Exception as e:
            logger.critical("Docker verification encountered an unexpected error: %s", e)
            self.has_gpu_runtime = False
            return False

    def build_device_requests(
        self,
        gpu_devices: Optional[Union[str, List[str]]] = None,
        gpu_count: Optional[int] = None,
        capabilities: Optional[List[List[str]]] = None,
    ) -> Optional[List[DeviceRequest]]:
        """
        Build Docker DeviceRequest list for GPU acceleration.

        Args:
            gpu_devices: "all", "none", comma-separated string ("0,1"), or list of device IDs (["0", "1"]).
            gpu_count: Specific number of GPUs to allocate (e.g. 1, 2).
            capabilities: Custom driver capabilities (defaults to [["gpu"]]).
        """
        caps = capabilities or [["gpu"]]

        # If explicitly disabled
        if gpu_devices == "none" or (isinstance(gpu_devices, str) and gpu_devices.lower() == "none"):
            return None

        # If specific device IDs are given
        if gpu_devices is not None and gpu_devices != "all":
            if isinstance(gpu_devices, str):
                device_ids = [d.strip() for d in gpu_devices.split(",") if d.strip()]
            elif isinstance(gpu_devices, (list, tuple)):
                device_ids = [str(d).strip() for d in gpu_devices if str(d).strip()]
            else:
                device_ids = [str(gpu_devices)]

            if device_ids:
                return [DeviceRequest(device_ids=device_ids, capabilities=caps)]

        # If specific GPU count is given
        if gpu_count is not None and gpu_count > 0:
            return [DeviceRequest(count=gpu_count, capabilities=caps)]

        # Default: all GPUs
        return [DeviceRequest(count=-1, capabilities=caps)]

    def pull_image(
        self,
        image: str,
        auth_config: Optional[dict] = None,
        timeout: int = 600,
    ):
        """Pull Docker image with progress logging."""
        logger.info("Pulling image: %s (timeout: %ds)", image, timeout)
        try:
            if auth_config:
                self.client.images.pull(image, auth_config=auth_config)
            else:
                self.client.images.pull(image)
            logger.info("Image pulled successfully: %s", image)
        except Exception as e:
            logger.error("Failed to pull image %s: %s", image, e)
            raise

    def run_job(
        self,
        job_id: str,
        image: str,
        command: Optional[Union[str, List[str]]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        input_data_url: Optional[str] = None,
        gpu_memory_mb: Optional[int] = None,
        gpu_devices: Optional[Union[str, List[str]]] = None,
        gpu_count: Optional[int] = None,
        use_gpu: bool = True,
        runtime: Optional[str] = None,
        cpu_limit: Optional[float] = None,
        memory_limit: Optional[str] = None,
        shm_size: Optional[str] = None,
        job_dir: Optional[str] = None,
        read_only: bool = True,
        tmpfs_size: str = "1G",
        network_mode: str = "bridge",
        ports: Optional[Dict[str, Any]] = None,
        pids_limit: Optional[int] = None,
        security_opt: Optional[List[str]] = None,
    ) -> str:
        """
        Run a compute job in a hardened, isolated Docker container.

        Args:
            job_id: Unique job identifier.
            image: Docker image name / tag.
            command: Command string or list to execute.
            env_vars: Environment variables to inject.
            input_data_url: URL to input dataset for environment injection.
            gpu_memory_mb: Target GPU VRAM allocation.
            gpu_devices: GPU selector ("all", "0,1", ["0"], etc.).
            gpu_count: Number of GPUs to allocate.
            use_gpu: Whether to request GPU acceleration.
            runtime: Docker runtime override (e.g. "nvidia").
            cpu_limit: Maximum CPU cores allocated (e.g. 4.0).
            memory_limit: RAM limit (e.g. "8g", "16g").
            shm_size: Shared memory size (e.g. "2g", "4g").
            job_dir: Base directory on host for job isolation.
            read_only: Mount container root filesystem as read-only.
            tmpfs_size: Size of tmpfs scratch memory mounted at /tmp.
            network_mode: Network configuration ("bridge", "none", "host").
            ports: Port mappings for SSH/Jupyter pass-through.
            pids_limit: Maximum PIDs limit to prevent fork bombs.
            security_opt: Extra security options (e.g. ["no-new-privileges:true"]).

        Returns:
            Container ID string.
        """
        job_dir = job_dir or f"/tmp/neuralops-jobs/{job_id}"
        workspace_dir = os.path.join(job_dir, "workspace")
        data_dir = os.path.join(workspace_dir, "data")
        artifacts_dir = os.path.join(workspace_dir, "artifacts")

        os.makedirs(workspace_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)

        # Build injected environment variables
        environment: Dict[str, str] = {
            "NEURALOPS_JOB_ID": job_id,
            "PYTHONUNBUFFERED": "1",
            "DEBIAN_FRONTEND": "noninteractive",
            "WORKSPACE": "/workspace",
            "DATA_DIR": "/workspace/data",
            "ARTIFACTS_DIR": "/workspace/artifacts",
        }

        if input_data_url:
            environment["INPUT_DATA_URL"] = input_data_url

        if env_vars:
            environment.update(env_vars)

        # Resource calculations
        allocated_cpus = cpu_limit if cpu_limit is not None else self.default_cpus
        nano_cpus = int(allocated_cpus * 1e9)
        allocated_mem = memory_limit or self.default_mem_limit
        allocated_shm = shm_size or self.default_shm_size
        allocated_pids = pids_limit if pids_limit is not None else self.default_pids_limit

        # Base container configuration
        container_config: Dict[str, Any] = {
            "image": image,
            "detach": True,
            "name": f"neuralops-job-{job_id[:12]}",
            "environment": environment,
            "volumes": {
                workspace_dir: {"bind": "/workspace", "mode": "rw"},
            },
            "working_dir": "/workspace",
            # Security & isolation
            "read_only": read_only,
            "tmpfs": {
                "/tmp": f"size={tmpfs_size},exec",
                "/run": "size=64M",
            },
            "network_mode": network_mode,
            "auto_remove": False,  # Retained to harvest exit codes and final logs
            # Resource constraints
            "mem_limit": allocated_mem,
            "nano_cpus": nano_cpus,
            "shm_size": allocated_shm,
            "pids_limit": allocated_pids,
        }

        if command:
            container_config["command"] = command

        if ports:
            container_config["ports"] = ports

        if security_opt:
            container_config["security_opt"] = security_opt

        # GPU acceleration configuration
        if use_gpu and self.has_gpu_runtime:
            device_requests = self.build_device_requests(
                gpu_devices=gpu_devices,
                gpu_count=gpu_count,
            )
            if device_requests:
                container_config["device_requests"] = device_requests
                # Ensure standard NVIDIA environment variables
                environment.setdefault("NVIDIA_VISIBLE_DEVICES", "all")
                environment.setdefault("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")
            if runtime:
                container_config["runtime"] = runtime
        elif runtime:
            container_config["runtime"] = runtime

        logger.info(
            "Launching container '%s' (image=%s, cpus=%.1f, mem=%s, gpu=%s)",
            container_config["name"],
            image,
            allocated_cpus,
            allocated_mem,
            "yes" if (use_gpu and self.has_gpu_runtime) else "no",
        )

        container = self.client.containers.run(**container_config)
        logger.info("Container %s started successfully for job %s", container.id[:12], job_id)
        return container.id

    def run_job_with_pre_script(
        self,
        job_id: str,
        image: str,
        command: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        pre_script: Optional[str] = None,
        input_data_url: Optional[str] = None,
        gpu_memory_mb: Optional[int] = None,
        gpu_devices: Optional[Union[str, List[str]]] = None,
        gpu_count: Optional[int] = None,
        use_gpu: bool = True,
        runtime: Optional[str] = None,
        cpu_limit: Optional[float] = None,
        memory_limit: Optional[str] = None,
        shm_size: Optional[str] = None,
        job_dir: Optional[str] = None,
        read_only: bool = True,
        tmpfs_size: str = "1G",
        network_mode: str = "bridge",
        ports: Optional[Dict[str, Any]] = None,
        pids_limit: Optional[int] = None,
        security_opt: Optional[List[str]] = None,
    ) -> str:
        """
        Run a container with a generated pre-script (e.g. data downloading, setup)
        chained with the primary user command.
        """
        job_dir = job_dir or f"/tmp/neuralops-jobs/{job_id}"
        workspace_dir = os.path.join(job_dir, "workspace")
        os.makedirs(workspace_dir, exist_ok=True)

        if pre_script:
            script_path = os.path.join(workspace_dir, "entrypoint.sh")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(pre_script)
                if command and command.strip():
                    f.write("\n\n# User Command\n")
                    f.write(command.strip() + "\n")

            os.chmod(script_path, 0o755)
            actual_command = ["/bin/bash", "/workspace/entrypoint.sh"]
        else:
            actual_command = command if command else ["/bin/bash"]

        return self.run_job(
            job_id=job_id,
            image=image,
            command=actual_command,
            env_vars=env_vars,
            input_data_url=input_data_url,
            gpu_memory_mb=gpu_memory_mb,
            gpu_devices=gpu_devices,
            gpu_count=gpu_count,
            use_gpu=use_gpu,
            runtime=runtime,
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
            shm_size=shm_size,
            job_dir=job_dir,
            read_only=read_only,
            tmpfs_size=tmpfs_size,
            network_mode=network_mode,
            ports=ports,
            pids_limit=pids_limit,
            security_opt=security_opt,
        )

    def get_container_status(self, container_id: str) -> str:
        """Get container status: running, exited, paused, not_found, error."""
        try:
            container = self.client.containers.get(container_id)
            return container.status
        except NotFound:
            return "not_found"
        except Exception as e:
            logger.warning("Error fetching status for container %s: %s", container_id, e)
            return "error"

    def get_container_info(self, container_id: str) -> Dict[str, Any]:
        """
        Get detailed container state information (status, exit code, OOM killed, timestamps).
        """
        try:
            container = self.client.containers.get(container_id)
            state = container.attrs.get("State", {})
            return {
                "id": container_id,
                "name": container.name,
                "status": container.status,
                "running": state.get("Running", False),
                "paused": state.get("Paused", False),
                "restarting": state.get("Restarting", False),
                "oom_killed": state.get("OOMKilled", False),
                "dead": state.get("Dead", False),
                "exit_code": state.get("ExitCode", 0),
                "error": state.get("Error", ""),
                "started_at": state.get("StartedAt", ""),
                "finished_at": state.get("FinishedAt", ""),
            }
        except NotFound:
            return {
                "id": container_id,
                "status": "not_found",
                "running": False,
                "exit_code": -1,
                "oom_killed": False,
                "error": "Container not found",
            }
        except Exception as e:
            return {
                "id": container_id,
                "status": "error",
                "running": False,
                "exit_code": -1,
                "oom_killed": False,
                "error": str(e),
            }

    def get_container_logs(
        self,
        container_id: str,
        tail: Union[int, str] = 100,
        timestamps: bool = False,
        stdout: bool = True,
        stderr: bool = True,
    ) -> str:
        """Get recent logs from a container as UTF-8 string."""
        try:
            container = self.client.containers.get(container_id)
            raw = container.logs(
                stdout=stdout,
                stderr=stderr,
                tail=tail if tail != 0 else "all",
                timestamps=timestamps,
            )
            return raw.decode("utf-8", errors="replace")
        except Exception as e:
            logger.debug("Failed to get logs for container %s: %s", container_id, e)
            return ""

    def stream_logs(
        self,
        container_id: str,
        follow: bool = True,
        tail: Union[int, str] = 100,
        since: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Stream container logs line-by-line or chunk-by-chunk in real time.
        """
        try:
            container = self.client.containers.get(container_id)
            stream_kwargs: Dict[str, Any] = {
                "stdout": True,
                "stderr": True,
                "stream": True,
                "follow": follow,
                "tail": tail if tail != 0 else "all",
            }
            if since is not None:
                stream_kwargs["since"] = since

            log_stream = container.logs(**stream_kwargs)
            for chunk in log_stream:
                if isinstance(chunk, bytes):
                    yield chunk.decode("utf-8", errors="replace")
                else:
                    yield str(chunk)
        except NotFound:
            return
        except Exception as e:
            logger.debug("Log streaming ended for %s: %s", container_id, e)
            return

    def stop_container(
        self,
        container_id: str,
        timeout: int = 10,
        remove: bool = True,
    ) -> bool:
        """
        Gracefully stop and optionally remove a container.
        """
        try:
            container = self.client.containers.get(container_id)
            logger.info("Stopping container %s (timeout: %ds)", container_id[:12], timeout)
            try:
                container.stop(timeout=timeout)
            except Exception as e:
                logger.warning("Stop container %s error, forcing kill: %s", container_id[:12], e)
                try:
                    container.kill()
                except Exception:
                    pass

            if remove:
                try:
                    container.remove(v=True, force=True)
                    logger.info("Container %s removed", container_id[:12])
                except Exception as rem_err:
                    logger.debug("Container removal note for %s: %s", container_id[:12], rem_err)
            return True
        except NotFound:
            return True
        except Exception as e:
            logger.error("Failed to stop/remove container %s: %s", container_id, e)
            return False

    def remove_container(self, container_id: str, force: bool = True, v: bool = True) -> bool:
        """Forcefully remove a container and its anonymous volumes."""
        try:
            container = self.client.containers.get(container_id)
            container.remove(force=force, v=v)
            return True
        except NotFound:
            return True
        except Exception as e:
            logger.error("Failed to remove container %s: %s", container_id, e)
            return False

    def cleanup(
        self,
        older_than_seconds: int = 3600,
        all_jobs: bool = False,
    ) -> int:
        """
        Clean up exited or stale NeuralOps containers.

        Returns:
            Number of containers removed.
        """
        removed_count = 0
        try:
            containers = self.client.containers.list(
                all=True, filters={"name": "neuralops-job-"}
            )
            now = time.time()
            for c in containers:
                try:
                    if all_jobs or c.status in ("exited", "dead"):
                        c.remove(force=True, v=True)
                        removed_count += 1
                except Exception as rem_err:
                    logger.debug("Failed removing container %s during cleanup: %s", c.id[:12], rem_err)
        except Exception as e:
            logger.warning("Container cleanup encountered an error: %s", e)

        return removed_count

    def exec_in_container(
        self,
        container_id: str,
        cmd: Union[str, List[str]],
        env: Optional[Dict[str, str]] = None,
        workdir: str = "/workspace",
    ) -> Tuple[int, str]:
        """Execute command inside running container."""
        try:
            container = self.client.containers.get(container_id)
            exec_res = container.exec_run(
                cmd=cmd,
                environment=env,
                workdir=workdir,
                stdout=True,
                stderr=True,
            )
            output = exec_res.output.decode("utf-8", errors="replace") if exec_res.output else ""
            return exec_res.exit_code, output
        except Exception as e:
            logger.error("Failed to exec in container %s: %s", container_id, e)
            return -1, str(e)
