"""
Remote distributed training – coordinate GPUs across the internet.
Handles NAT, authentication, and job dispatch.
"""

from .coordinator import RemoteCoordinator
from .worker_agent import RemoteWorkerAgent
from .rendezvous import InternetRendezvous
from .integration import launch_remote_training

__all__ = [
    "RemoteCoordinator",
    "RemoteWorkerAgent",
    "InternetRendezvous",
    "launch_remote_training",
]