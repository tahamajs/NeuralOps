"""
Cluster Manager – lightweight cluster management without SLURM/K8s.

Inspired by TorchSubmit: SSH‑based node management, code sync, and
torchrun‑based job execution across arbitrary machines.

Reference:
  - TorchSubmit: https://github.com/ParthaEth/torch-submit
"""

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
    """Description of a machine in the cluster."""

    hostname: str
    public_ip: str
    private_ip: str
    ssh_user: str = "ubuntu"
    ssh_key_path: str = "~/.ssh/id_rsa"
    ssh_port: int = 22
    gpu_count: int = 0
    role: NodeRole = NodeRole.WORKER


class ClusterManager:
    """
    Manages a set of GPU machines over SSH – no SLURM or Kubernetes needed.

    Usage:
        cluster = ClusterManager()
        cluster.add_node("gpu1.example.com", "192.168.1.10")
        cluster.sync_code(local_dir=".", remote_dir="~/workspace")
        cluster.launch_training("python train.py --epochs 10",
                                nproc_per_node=4, nnodes=2)
    """

    def __init__(self) -> None:
        self._nodes: List[NodeSpec] = []
        self._master: Optional[NodeSpec] = None

    def add_node(
        self,
        public_ip: str,
        private_ip: str = "",
        hostname: str = "",
        ssh_user: str = "ubuntu",
        ssh_key_path: str = "~/.ssh/id_rsa",
        role: NodeRole = NodeRole.WORKER,
    ) -> None:
        """Register a machine in the cluster."""
        node = NodeSpec(
            hostname=hostname or public_ip,
            public_ip=public_ip,
            private_ip=private_ip or public_ip,
            ssh_user=ssh_user,
            ssh_key_path=ssh_key_path,
            role=role,
        )
        self._nodes.append(node)
        if role == NodeRole.MASTER:
            self._master = node
        logger.info("Added node: %s (%s)", node.hostname, node.public_ip)

    def sync_code(self, local_dir: str, remote_dir: str) -> None:
        """Copy local code to all worker nodes using rsync."""
        for node in self._nodes:
            self._ssh_run(
                node,
                f"mkdir -p {remote_dir}",
                quiet=True,
            )
            cmd = (
                f"rsync -avz --exclude '__pycache__' --exclude '.git' "
                f"-e 'ssh -i {node.ssh_key_path} -p {node.ssh_port}' "
                f"{local_dir}/ {node.ssh_user}@{node.public_ip}:{remote_dir}/"
            )
            subprocess.run(cmd, shell=True, check=True)
            logger.info("Synced code to %s", node.hostname)

    def launch_training(
        self,
        command: str,
        nproc_per_node: int = 1,
        nnodes: int = 0,
        working_dir: str = "~/workspace",
    ) -> None:
        """
        Launch torchrun across all nodes.

        Args:
            command: the Python command to run (e.g., "python train.py")
            nproc_per_node: number of processes per node (usually GPU count)
            nnodes: number of nodes (0 = use all registered nodes)
        """
        if nnodes == 0:
            nnodes = len(self._nodes)
        if nnodes > len(self._nodes):
            raise ValueError(
                f"Requested {nnodes} nodes, only {len(self._nodes)} registered"
            )

        nodes = self._nodes[:nnodes]
        master = nodes[0]

        # Determine private IP for NCCL communication
        master_addr = master.private_ip or master.public_ip
        master_port = 29500

        processes = []
        for rank, node in enumerate(nodes):
            env_vars = (
                f"export MASTER_ADDR={master_addr} && "
                f"export MASTER_PORT={master_port} && "
                f"export WORLD_SIZE={nnodes * nproc_per_node} && "
                f"export RANK={rank * nproc_per_node} && "
                f"export NODE_RANK={rank} && "
                f"export NCCL_SOCKET_IFNAME=eth0"
            )
            full_cmd = f"cd {working_dir} && {env_vars} && torchrun --nproc_per_node={nproc_per_node} --nnodes={nnodes} --node_rank={rank} --master_addr={master_addr} --master_port={master_port} {command}"

            proc = threading.Thread(
                target=self._ssh_run,
                args=(node, full_cmd),
                daemon=True,
            )
            proc.start()
            processes.append(proc)

        logger.info(
            "Launched training on %d nodes (%d GPUs each)", nnodes, nproc_per_node
        )

        # Wait for all to finish
        for proc in processes:
            proc.join()

    def check_gpu_availability(self) -> Dict[str, int]:
        """Query GPU count on each node."""
        result = {}
        for node in self._nodes:
            out = self._ssh_run(
                node,
                "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l",
                capture=True,
            )
            result[node.hostname] = int(out.strip())
        return result

    def _ssh_run(
        self, node: NodeSpec, command: str, quiet: bool = False, capture: bool = False
    ) -> Optional[str]:
        """Execute a command on a remote node via SSH."""
        ssh_cmd = (
            f"ssh -i {node.ssh_key_path} -p {node.ssh_port} "
            f"-o StrictHostKeyChecking=no "
            f"{node.ssh_user}@{node.public_ip} '{command}'"
        )
        if capture:
            return subprocess.check_output(ssh_cmd, shell=True).decode()
        if quiet:
            subprocess.run(ssh_cmd, shell=True, capture_output=True)
        else:
            subprocess.run(ssh_cmd, shell=True, check=True)
        return None
