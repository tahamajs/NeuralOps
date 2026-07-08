"""
Remote distributed training – coordinate GPUs across the internet.
Handles NAT, authentication, and job dispatch.
"""

from .coordinator import RemoteCoordinator
from .integration import launch_remote_training
from .rendezvous import InternetRendezvous
from .worker_agent import RemoteWorkerAgent

__all__ = [
    "RemoteCoordinator",
    "RemoteWorkerAgent",
    "InternetRendezvous",
    "launch_remote_training",
]
