"""
InternetRendezvous – helps GPUs behind NAT find each other using
a central coordinating server (similar to etcd or PyTorch Elastic store).
"""

import time
from typing import Dict, Optional

import requests


class InternetRendezvous:
    """
    A simple key‑value store on the central Django server that workers
    use to exchange their actual reachable addresses.
    """

    def __init__(self, server_url: str, job_id: str, auth_token: str) -> None:
        self.server_url = server_url
        self.job_id = job_id
        self.auth_token = auth_token

    def set_address(self, rank: int, host: str, port: int) -> bool:
        """Announce this worker's reachable address."""
        try:
            resp = requests.post(
                f"{self.server_url}/api/remote/rendezvous/set/",
                json={
                    "job_id": self.job_id,
                    "rank": rank,
                    "host": host,
                    "port": port,
                    "auth_token": self.auth_token,
                },
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def wait_for_peers(
        self, rank: int, world_size: int, timeout: int = 300
    ) -> Optional[Dict[int, tuple]]:
        """Wait until all peers have announced their addresses."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = requests.get(
                    f"{self.server_url}/api/remote/rendezvous/get/",
                    params={
                        "job_id": self.job_id,
                        "auth_token": self.auth_token,
                    },
                    timeout=5,
                )
                if resp.status_code == 200:
                    peers = resp.json()
                    if len(peers) == world_size:
                        return {
                            int(k): (v["host"], v["port"]) for k, v in peers.items()
                        }
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
                pass
            time.sleep(2)
        return None
