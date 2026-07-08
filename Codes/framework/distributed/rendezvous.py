"""Internet rendezvous — helps GPUs behind NAT find each other."""

import logging
import time
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)


class InternetRendezvous:
    """Key‑value store on a central server for worker address exchange."""

    def __init__(self, server_url: str, job_id: str, auth_token: str):
        self.server_url = server_url
        self.job_id = job_id
        self.auth_token = auth_token

    def set_address(self, rank: int, host: str, port: int) -> bool:
        try:
            r = requests.post(
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
            return r.status_code == 200
        except:
            return False

    def wait_for_peers(
        self, rank: int, world_size: int, timeout: int = 300
    ) -> Optional[Dict[int, tuple]]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(
                    f"{self.server_url}/api/remote/rendezvous/get/",
                    params={"job_id": self.job_id, "auth_token": self.auth_token},
                    timeout=5,
                )
                if r.status_code == 200:
                    peers = r.json()
                    if len(peers) == world_size:
                        return {
                            int(k): (v["host"], v["port"]) for k, v in peers.items()
                        }
            except:
                pass
            time.sleep(2)
        return None
