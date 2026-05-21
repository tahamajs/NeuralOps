"""
Container Controller — pause / resume / stop / restart / logs for Docker containers
that are running compute jobs.  Used by the provider GUI so the provider has
full control over what's happening on their GPU.
"""
import logging, time
import docker
from typing import Optional

logger = logging.getLogger("neuralops.container_ctrl")

class ContainerController:
    """Thin wrapper around docker‑py for job container management."""

    def __init__(self):
        self._client: Optional[docker.DockerClient] = None

    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    # ── Lifecycle ────────────────────────────────────────────────
    def pause(self, container_id: str) -> dict:
        """Pause (freeze cgroup) a container."""
        try:
            c = self.client.containers.get(container_id)
            c.pause()
            return {'status': 'paused', 'container_id': container_id}
        except docker.errors.NotFound:
            return {'status': 'not_found', 'container_id': container_id}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def unpause(self, container_id: str) -> dict:
        """Resume a paused container."""
        try:
            c = self.client.containers.get(container_id)
            c.unpause()
            return {'status': 'running', 'container_id': container_id}
        except docker.errors.NotFound:
            return {'status': 'not_found', 'container_id': container_id}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def stop(self, container_id: str, timeout: int = 10) -> dict:
        """Stop a container gracefully."""
        try:
            c = self.client.containers.get(container_id)
            c.stop(timeout=timeout)
            return {'status': 'stopped', 'container_id': container_id}
        except docker.errors.NotFound:
            return {'status': 'not_found', 'container_id': container_id}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def restart(self, container_id: str, timeout: int = 10) -> dict:
        try:
            c = self.client.containers.get(container_id)
            c.restart(timeout=timeout)
            return {'status': 'restarted', 'container_id': container_id}
        except docker.errors.NotFound:
            return {'status': 'not_found', 'container_id': container_id}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    # ── Info ─────────────────────────────────────────────────────
    def status(self, container_id: str) -> dict:
        try:
            c = self.client.containers.get(container_id)
            return {
                'container_id': container_id,
                'status': c.status,
                'name': c.name,
                'image': c.image.tags[0] if c.image.tags else 'unknown',
                'created': c.attrs.get('Created', ''),
                'state': c.attrs.get('State', {}),
            }
        except docker.errors.NotFound:
            return {'status': 'not_found', 'container_id': container_id}

    def stats(self, container_id: str) -> dict:
        """Return a single stats snapshot (like docker stats)."""
        try:
            c = self.client.containers.get(container_id)
            # stats(stream=False) returns one snapshot
            raw = c.stats(stream=False)
            cpu_delta = raw['cpu_stats']['cpu_usage']['total_usage'] - raw['precpu_stats']['cpu_usage']['total_usage']
            system_delta = raw['cpu_stats']['system_cpu_usage'] - raw['precpu_stats']['system_cpu_usage']
            cpu_pct = (cpu_delta / system_delta * len(raw['cpu_stats']['cpu_usage'].get('percpu_usage', [1]))) * 100 if system_delta > 0 else 0
            mem = raw['memory_stats']
            mem_used = mem.get('usage', 0) - mem.get('stats', {}).get('cache', 0)
            mem_limit = mem.get('limit', 1)
            return {
                'container_id': container_id,
                'cpu_percent': round(cpu_pct, 1),
                'memory_used_mb': round(mem_used / (1024 * 1024), 1),
                'memory_limit_mb': round(mem_limit / (1024 * 1024), 1),
                'memory_pct': round(mem_used / max(mem_limit, 1) * 100, 1),
                'network_rx_bytes': sum(v.get('rx_bytes', 0) for v in raw.get('networks', {}).values()),
                'network_tx_bytes': sum(v.get('tx_bytes', 0) for v in raw.get('networks', {}).values()),
            }
        except docker.errors.NotFound:
            return {'status': 'not_found'}
        except Exception as e:
            logger.warning("Container stats failed: %s", e)
            return {'status': 'error', 'error': str(e)}

    def logs(self, container_id: str, tail: int = 200) -> str:
        try:
            c = self.client.containers.get(container_id)
            return c.logs(tail=tail).decode('utf-8', errors='replace')
        except docker.errors.NotFound:
            return ''
        except Exception as e:
            return f'Error: {e}'