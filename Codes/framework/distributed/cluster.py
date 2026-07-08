"""Lightweight SSH‑based cluster management — no SLURM needed."""

import logging
import subprocess
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class NodeRole(Enum):
    MASTER = "master"
    WORKER = "worker"


@dataclass
class NodeSpec:
    hostname: str
    public_ip: str
    private_ip: str = ""
    ssh_user: str = "ubuntu"
    ssh_key_path: str = "~/.ssh/id_rsa"
    ssh_port: int = 22
    gpu_count: int = 0
    role: NodeRole = NodeRole.WORKER


class ClusterManager:
    """Manages GPU machines over SSH — sync code, launch torchrun."""

    def __init__(self):
        self._nodes: List[NodeSpec] = []
        self._master: Optional[NodeSpec] = None

    def add_node(
        self,
        public_ip: str,
        private_ip: str = "",
        hostname: str = "",
        ssh_user: str = "ubuntu",
        ssh_key: str = "~/.ssh/id_rsa",
        role: NodeRole = NodeRole.WORKER,
        gpu_count: int = 0,
    ):
        node = NodeSpec(
            hostname=hostname or public_ip,
            public_ip=public_ip,
            private_ip=private_ip or public_ip,
            ssh_user=ssh_user,
            ssh_key_path=ssh_key,
            role=role,
            gpu_count=gpu_count,
        )
        self._nodes.append(node)
        if role == NodeRole.MASTER:
            self._master = node
        logger.info("Added node: %s", node.hostname)

    def sync_code(self, local_dir: str, remote_dir: str = "~/workspace"):
        for node in self._nodes:
            self._run(node, f"mkdir -p {remote_dir}", quiet=True)
            subprocess.run(
                [
                    "rsync",
                    "-avz",
                    "--exclude=__pycache__",
                    "--exclude=.git",
                    "-e",
                    f"ssh -i {node.ssh_key_path} -p {node.ssh_port}",
                    f"{local_dir}/",
                    f"{node.ssh_user}@{node.public_ip}:{remote_dir}/",
                ],
                check=True,
            )
        logger.info("Code synced to all nodes.")

    def launch(
        self,
        command: str,
        nproc_per_node: int = 1,
        nnodes: int = 0,
        working_dir: str = "~/workspace",
    ):
        if nnodes == 0:
            nnodes = len(self._nodes)
        nodes = self._nodes[:nnodes]
        master = nodes[0]
        master_addr = master.private_ip or master.public_ip

        processes = []
        for rank, node in enumerate(nodes):
            env = (
                f"export MASTER_ADDR={master_addr} && export MASTER_PORT=29500 && "
                f"export WORLD_SIZE={nnodes * nproc_per_node} && "
                f"export NODE_RANK={rank} && export NCCL_SOCKET_IFNAME=eth0"
            )
            full = (
                f"cd {working_dir} && {env} && "
                f"torchrun --nproc_per_node={nproc_per_node} --nnodes={nnodes} "
                f"--node_rank={rank} --master_addr={master_addr} --master_port=29500 "
                f"{command}"
            )
            t = threading.Thread(target=self._run, args=(node, full), daemon=True)
            t.start()
            processes.append(t)
        for t in processes:
            t.join()
        logger.info("Training finished on %d nodes.", nnodes)

    def check_gpus(self) -> Dict[str, int]:
        result = {}
        for node in self._nodes:
            out = self._run(
                node,
                "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l",
                capture=True,
            )
            result[node.hostname] = int(out.strip())
        return result

    def _run(
        self, node: NodeSpec, cmd: str, quiet: bool = False, capture: bool = False
    ) -> Optional[str]:
        ssh = (
            f"ssh -i {node.ssh_key_path} -p {node.ssh_port} "
            f"-o StrictHostKeyChecking=no {node.ssh_user}@{node.public_ip} '{cmd}'"
        )
        if capture:
            return subprocess.check_output(ssh, shell=True).decode()
        subprocess.run(ssh, shell=True, check=not quiet, capture_output=quiet)
        return None
