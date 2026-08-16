"""
Container Controller — pause / resume / stop / restart / kill / stats / logs
for Docker containers executing compute jobs.

Used by the provider GUI so the provider has full real-time visibility
and granular control over GPU and container execution.
"""

import logging
from typing import Any, Dict, Generator, List, Optional, Union

import docker
from docker.errors import APIError, DockerException, NotFound

logger = logging.getLogger("neuralops.container_ctrl")


class ContainerController:
    """Thin wrapper around docker-py for job container management and metrics."""

    def __init__(self, client: Optional[docker.DockerClient] = None):
        self._client: Optional[docker.DockerClient] = client

    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    # ── Lifecycle Actions ─────────────────────────────────────────

    def pause(self, container_id: str) -> Dict[str, Any]:
        """Pause (freeze cgroup) a container."""
        try:
            c = self.client.containers.get(container_id)
            c.pause()
            logger.info("Paused container %s", container_id[:12])
            return {"status": "paused", "container_id": container_id}
        except NotFound:
            return {"status": "not_found", "container_id": container_id, "error": "Container not found"}
        except Exception as e:
            logger.error("Failed to pause container %s: %s", container_id, e)
            return {"status": "error", "container_id": container_id, "error": str(e)}

    def unpause(self, container_id: str) -> Dict[str, Any]:
        """Resume a paused container."""
        try:
            c = self.client.containers.get(container_id)
            c.unpause()
            logger.info("Resumed container %s", container_id[:12])
            return {"status": "running", "container_id": container_id}
        except NotFound:
            return {"status": "not_found", "container_id": container_id, "error": "Container not found"}
        except Exception as e:
            logger.error("Failed to unpause container %s: %s", container_id, e)
            return {"status": "error", "container_id": container_id, "error": str(e)}

    def stop(self, container_id: str, timeout: int = 10) -> Dict[str, Any]:
        """Stop a container gracefully."""
        try:
            c = self.client.containers.get(container_id)
            c.stop(timeout=timeout)
            logger.info("Stopped container %s", container_id[:12])
            return {"status": "stopped", "container_id": container_id}
        except NotFound:
            return {"status": "not_found", "container_id": container_id, "error": "Container not found"}
        except Exception as e:
            logger.error("Failed to stop container %s: %s", container_id, e)
            return {"status": "error", "container_id": container_id, "error": str(e)}

    def restart(self, container_id: str, timeout: int = 10) -> Dict[str, Any]:
        """Restart a container."""
        try:
            c = self.client.containers.get(container_id)
            c.restart(timeout=timeout)
            logger.info("Restarted container %s", container_id[:12])
            return {"status": "restarted", "container_id": container_id}
        except NotFound:
            return {"status": "not_found", "container_id": container_id, "error": "Container not found"}
        except Exception as e:
            logger.error("Failed to restart container %s: %s", container_id, e)
            return {"status": "error", "container_id": container_id, "error": str(e)}

    def kill(self, container_id: str, signal: str = "SIGKILL") -> Dict[str, Any]:
        """Send a termination signal to a container."""
        try:
            c = self.client.containers.get(container_id)
            c.kill(signal=signal)
            logger.info("Sent %s to container %s", signal, container_id[:12])
            return {"status": "killed", "container_id": container_id, "signal": signal}
        except NotFound:
            return {"status": "not_found", "container_id": container_id, "error": "Container not found"}
        except Exception as e:
            logger.error("Failed to kill container %s: %s", container_id, e)
            return {"status": "error", "container_id": container_id, "error": str(e)}

    # ── Info & Metrics ────────────────────────────────────────────

    def status(self, container_id: str) -> Dict[str, Any]:
        """Get current container status and detailed metadata."""
        try:
            c = self.client.containers.get(container_id)
            state = c.attrs.get("State", {})
            return {
                "container_id": container_id,
                "status": c.status,
                "name": c.name,
                "image": c.image.tags[0] if (c.image and c.image.tags) else (c.attrs.get("Config", {}).get("Image", "unknown")),
                "created": c.attrs.get("Created", ""),
                "state": state,
                "exit_code": state.get("ExitCode", 0),
                "oom_killed": state.get("OOMKilled", False),
                "started_at": state.get("StartedAt", ""),
                "finished_at": state.get("FinishedAt", ""),
            }
        except NotFound:
            return {"container_id": container_id, "status": "not_found", "error": "Container not found"}
        except Exception as e:
            logger.error("Status inspection error for %s: %s", container_id, e)
            return {"container_id": container_id, "status": "error", "error": str(e)}

    def stats(self, container_id: str) -> Dict[str, Any]:
        """
        Return a single stats snapshot (CPU %, Memory, Network, Block I/O).
        Compatible with both cgroup v1 and cgroup v2 Docker daemon setups.
        """
        try:
            c = self.client.containers.get(container_id)
            raw = c.stats(stream=False)

            if not isinstance(raw, dict):
                return {"status": "error", "error": "Invalid stats response"}

            # CPU percentage calculation
            cpu_stats = raw.get("cpu_stats", {})
            precpu_stats = raw.get("precpu_stats", {})

            cpu_usage_total = cpu_stats.get("cpu_usage", {}).get("total_usage", 0)
            precpu_usage_total = precpu_stats.get("cpu_usage", {}).get("total_usage", 0)
            cpu_delta = cpu_usage_total - precpu_usage_total

            system_cpu_usage = cpu_stats.get("system_cpu_usage", 0)
            presystem_cpu_usage = precpu_stats.get("system_cpu_usage", 0)
            system_delta = system_cpu_usage - presystem_cpu_usage

            online_cpus = cpu_stats.get("online_cpus")
            if not online_cpus:
                percpu_usage = cpu_stats.get("cpu_usage", {}).get("percpu_usage", [])
                online_cpus = len(percpu_usage) if percpu_usage else 1

            if system_delta > 0 and cpu_delta > 0:
                cpu_pct = (cpu_delta / system_delta) * online_cpus * 100.0
            else:
                cpu_pct = 0.0

            # Memory metrics
            mem_stats = raw.get("memory_stats", {})
            mem_usage_raw = mem_stats.get("usage", 0)
            stats_sub = mem_stats.get("stats", {})
            # cgroup v1 cache / cgroup v2 inactive_file
            cache_sub = stats_sub.get("cache", stats_sub.get("inactive_file", 0))
            mem_used = max(0, mem_usage_raw - cache_sub)
            mem_limit = mem_stats.get("limit", 1)
            mem_pct = (mem_used / mem_limit * 100.0) if mem_limit > 0 else 0.0

            # Network I/O
            networks = raw.get("networks", {})
            rx_bytes = sum(v.get("rx_bytes", 0) for v in networks.values())
            tx_bytes = sum(v.get("tx_bytes", 0) for v in networks.values())

            # Block I/O
            blkio_stats = raw.get("blkio_stats", {})
            io_service_bytes = blkio_stats.get("io_service_bytes_recursive", [])
            io_read_bytes = 0
            io_write_bytes = 0
            if io_service_bytes:
                for entry in io_service_bytes:
                    op = entry.get("op", "").lower()
                    if op == "read":
                        io_read_bytes += entry.get("value", 0)
                    elif op == "write":
                        io_write_bytes += entry.get("value", 0)

            # Pids
            pids_current = raw.get("pids_stats", {}).get("current", 0)

            return {
                "container_id": container_id,
                "cpu_percent": round(cpu_pct, 2),
                "memory_used_mb": round(mem_used / (1024 * 1024), 2),
                "memory_limit_mb": round(mem_limit / (1024 * 1024), 2),
                "memory_pct": round(mem_pct, 2),
                "network_rx_bytes": rx_bytes,
                "network_tx_bytes": tx_bytes,
                "block_read_bytes": io_read_bytes,
                "block_write_bytes": io_write_bytes,
                "pids_count": pids_current,
                "status": "active",
            }
        except NotFound:
            return {"status": "not_found", "container_id": container_id}
        except Exception as e:
            logger.warning("Container stats query failed for %s: %s", container_id, e)
            return {"status": "error", "container_id": container_id, "error": str(e)}

    def logs(self, container_id: str, tail: Union[int, str] = 200) -> str:
        """Fetch latest container logs as string."""
        try:
            c = self.client.containers.get(container_id)
            raw = c.logs(tail=tail if tail != 0 else "all")
            return raw.decode("utf-8", errors="replace")
        except NotFound:
            return ""
        except Exception as e:
            logger.debug("Error reading logs for %s: %s", container_id, e)
            return f"Error: {e}"

    def stream_logs(
        self, container_id: str, tail: Union[int, str] = 100, follow: bool = True
    ) -> Generator[str, None, None]:
        """Stream container logs line-by-line."""
        try:
            c = self.client.containers.get(container_id)
            for chunk in c.logs(
                stream=True,
                follow=follow,
                tail=tail if tail != 0 else "all",
                stdout=True,
                stderr=True,
            ):
                if isinstance(chunk, bytes):
                    yield chunk.decode("utf-8", errors="replace")
                else:
                    yield str(chunk)
        except NotFound:
            return
        except Exception as e:
            logger.debug("Log streaming error for %s: %s", container_id, e)
            return

    def exec_in_container(
        self,
        container_id: str,
        cmd: Union[str, List[str]],
        env: Optional[Dict[str, str]] = None,
        workdir: str = "/workspace",
    ) -> Dict[str, Any]:
        """Execute command in running container and return result dictionary."""
        try:
            c = self.client.containers.get(container_id)
            res = c.exec_run(
                cmd=cmd,
                environment=env,
                workdir=workdir,
                stdout=True,
                stderr=True,
            )
            output = res.output.decode("utf-8", errors="replace") if res.output else ""
            return {
                "status": "success",
                "container_id": container_id,
                "exit_code": res.exit_code,
                "output": output,
            }
        except NotFound:
            return {
                "status": "not_found",
                "container_id": container_id,
                "error": "Container not found",
            }
        except Exception as e:
            return {
                "status": "error",
                "container_id": container_id,
                "error": str(e),
            }

    def list_job_containers(
        self, all_containers: bool = True, prefix: str = "neuralops-job-"
    ) -> List[Dict[str, Any]]:
        """List all NeuralOps job containers."""
        results = []
        try:
            containers = self.client.containers.list(all=all_containers)
            for c in containers:
                if c.name.startswith(prefix):
                    results.append(
                        {
                            "id": c.id,
                            "name": c.name,
                            "status": c.status,
                            "image": c.image.tags[0] if (c.image and c.image.tags) else "unknown",
                            "created": c.attrs.get("Created", ""),
                        }
                    )
        except Exception as e:
            logger.error("Failed to list job containers: %s", e)
        return results
