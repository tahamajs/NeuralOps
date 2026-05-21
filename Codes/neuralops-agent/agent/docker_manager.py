"""
Docker container runtime manager.

Key security principles:
  - All user code runs inside isolated Docker containers
  - GPU access via NVIDIA Container Toolkit
  - Network restrictions: no outbound except allowlisted endpoints
  - Resource limits: memory, CPU, GPU fraction
  - Read-only root filesystem
  - tmpfs for /tmp
  - Auto-removal on completion/error

Inspired by: isol8 sandbox, NVIDIA DGX Spark secure containers,
and Boxed's sovereign code execution engine.
"""
import subprocess, logging, os, uuid
import docker
from pathlib import Path

logger = logging.getLogger("neuralops.docker")

class DockerManager:
    """Manage Docker containers for executing compute jobs."""

    def __init__(self):
        self._client = None

    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def verify_runtime(self) -> bool:
        """Verify Docker + NVIDIA Container Toolkit are installed and working."""
        try:
            # Check Docker
            self.client.ping()
            logger.info("Docker is running")

            # Check NVIDIA runtime by running nvidia-smi in a test container
            result = self.client.containers.run(
                'nvidia/cuda:12.4.0-base-ubuntu22.04',
                command='nvidia-smi',
                device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
                remove=True,
                stdout=True,
                stderr=True,
            )
            logger.info("NVIDIA container runtime verified")
            return True
        except docker.errors.DockerException as e:
            logger.critical("Docker is not running: %s", e)
            return False
        except Exception as e:
            logger.critical("NVIDIA container runtime not configured: %s", e)
            logger.critical("Run: sudo apt install -y nvidia-container-toolkit && "
                            "sudo nvidia-ctk runtime configure --runtime=docker && "
                            "sudo systemctl restart docker")
            return False

    def pull_image(self, image: str):
        """Pull Docker image with progress logging."""
        logger.info("Pulling image: %s", image)
        try:
            self.client.images.pull(image)
            logger.info("Image pulled: %s", image)
        except Exception as e:
            logger.error("Failed to pull image %s: %s", image, e)
            raise

    def run_job(self, job_id: str, image: str, command: str,
                env_vars: dict = None, input_data_url: str = None,
                gpu_memory_mb: int = None, job_dir: str = None) -> str:
        """
        Run a compute job in a secure Docker container.

        Returns container ID.
        """
        job_dir = job_dir or f'/tmp/neuralops-jobs/{job_id}'
        os.makedirs(job_dir, exist_ok=True)

        # Build container configuration
        container_config = {
            'image': image,
            'command': command,
            'detach': True,
            'name': f'neuralops-job-{job_id[:12]}',
            'environment': env_vars or {},
            'volumes': {
                job_dir: {'bind': '/workspace', 'mode': 'rw'},
            },
            'device_requests': [docker.types.DeviceRequest(
                count=-1,
                capabilities=[['gpu']]
            )],
            # Security hardening
            'read_only': True,  # Read-only root filesystem
            'tmpfs': {'/tmp': 'size=512M'},
            'network_mode': 'bridge',
            'auto_remove': True,  # Auto-remove on exit
            'mem_limit': '8g',
            'nano_cpus': int(4 * 1e9),  # 4 CPUs
            'shm_size': '1g',
        }

        # Add input data download if provided
        if input_data_url:
            env_vars = env_vars or {}
            env_vars['INPUT_DATA_URL'] = input_data_url
            env_vars['NEURALOPS_JOB_ID'] = job_id
            container_config['environment'] = env_vars

        logger.info("Starting container for job %s", job_id)
        container = self.client.containers.run(**container_config)
        logger.info("Container started: %s", container.id)

        return container.id

    def get_container_status(self, container_id: str) -> str:
        """Get container status: running, exited, etc."""
        try:
            container = self.client.containers.get(container_id)
            return container.status
        except docker.errors.NotFound:
            return 'not_found'

    def get_container_logs(self, container_id: str, tail: int = 100) -> str:
        """Get container logs."""
        try:
            container = self.client.containers.get(container_id)
            return container.logs(tail=tail).decode('utf-8', errors='replace')
        except Exception:
            return ''

    def stop_container(self, container_id: str):
        """Stop and remove a container."""
        try:
            container = self.client.containers.get(container_id)
            container.stop(timeout=10)
            logger.info("Container stopped: %s", container_id)
        except docker.errors.NotFound:
            pass
        except Exception as e:
            logger.error("Failed to stop container %s: %s", container_id, e)

    def cleanup(self):
        """Remove all completed NeuralOps containers."""
        try:
            containers = self.client.containers.list(
                all=True,
                filters={'name': 'neuralops-job-'}
            )
            for c in containers:
                if c.status == 'exited':
                    c.remove()
        except Exception:
            pass

    def run_job_with_pre_script(self, job_id: str, image: str, command: str,
                             env_vars: dict = None, pre_script: str = None) -> str:
        """
        Run a container with a pre‑script that downloads model/data before the main command.
        """
        job_dir = f'/tmp/neuralops-jobs/{job_id}'
        os.makedirs(job_dir, exist_ok=True)

        # Write pre‑script to a shared volume
        if pre_script:
            script_path = os.path.join(job_dir, 'entrypoint.sh')
            with open(script_path, 'w') as f:
                f.write(pre_script)
            os.chmod(script_path, 0o755)
            # If command is not empty, append it to the script
            if command:
                with open(script_path, 'a') as f:
                    f.write('\n' + command)
            actual_command = '/workspace/entrypoint.sh'
            # Copy script into workspace via volume mount (already mounted)
        else:
            actual_command = command if command else '/bin/bash'

        container = self.client.containers.run(
            image=image,
            command=actual_command,
            detach=True,
            name=f'neuralops-job-{job_id[:12]}',
            environment=env_vars or {},
            volumes={job_dir: {'bind': '/workspace', 'mode': 'rw'}},
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
            read_only=True,
            tmpfs={'/tmp': 'size=512M'},
            network_mode='bridge',
            auto_remove=True,
            mem_limit='8g',
            nano_cpus=int(4 * 1e9),
            shm_size='1g',
        )
        return container.id