"""Agent configuration from YAML file + CLI args + env vars."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class AgentConfig:
    api_key: str = ""
    api_base_url: str = "https://api.neuralops.com"
    ws_url: str = "wss://api.neuralops.com/ws/provider/"
    listing_id: str = ""
    version: str = "1.0.0"

    # Provider settings
    base_price_per_hour: float = 0.50
    listing_type: str = "on_demand"  # on_demand | spot | reserved
    location: str = ""
    storage_gb: int = 100
    bandwidth_mbps: int = 1000
    supports_ssh: bool = True
    supports_jupyter: bool = False

    # Resource limits
    max_concurrent_jobs: int = 2
    gpu_memory_fraction: float = 0.90  # leave 10% for system

    # Network & Heartbeat
    speed_test_interval_hours: int = 24
    heartbeat_interval_seconds: int = 10  # 10s heartbeat requirement
    heartbeat_timeout_seconds: int = 5
    max_missed_pings: int = 3

    # Reconnection & Backoff
    reconnect_initial_backoff: float = 1.0
    reconnect_max_backoff: float = 60.0
    reconnect_backoff_factor: float = 2.0
    reconnect_jitter: bool = True

    # Auth & Token Rotation
    auto_token_rotation: bool = True
    token_rotation_interval_seconds: int = 86400  # 24 hours

    # Paths
    data_dir: str = str(Path.home() / ".neuralops-agent")
    job_dir: str = str(Path.home() / ".neuralops-agent" / "jobs")
    log_dir: str = str(Path.home() / ".neuralops-agent" / "logs")

    # Auto-update
    auto_update: bool = True
    update_channel: str = "stable"

    # Logging
    log_level: str = "INFO"
    log_file: str = ""

    # Extra options / metadata
    extra: Dict[str, Any] = field(default_factory=dict)

    def update_from_dict(self, data: Dict[str, Any]) -> "AgentConfig":
        """Update configuration from flat or nested dictionary."""
        if not isinstance(data, dict):
            return self

        # Handle top-level keys
        for k, v in data.items():
            if hasattr(self, k) and k != "extra":
                setattr(self, k, v)

        # Handle nested sections (matches agent-config.yaml structure)
        if "listing" in data and isinstance(data["listing"], dict):
            for k, v in data["listing"].items():
                if hasattr(self, k):
                    setattr(self, k, v)

        if "resources" in data and isinstance(data["resources"], dict):
            for k, v in data["resources"].items():
                if hasattr(self, k):
                    setattr(self, k, v)

        if "network" in data and isinstance(data["network"], dict):
            for k, v in data["network"].items():
                if hasattr(self, k):
                    setattr(self, k, v)

        if "paths" in data and isinstance(data["paths"], dict):
            paths = data["paths"]
            if "data_dir" in paths:
                self.data_dir = os.path.expanduser(str(paths["data_dir"]))
            if "job_dir" in paths:
                self.job_dir = os.path.expanduser(str(paths["job_dir"]))
            if "log_dir" in paths:
                self.log_dir = os.path.expanduser(str(paths["log_dir"]))

        if "updates" in data and isinstance(data["updates"], dict):
            updates = data["updates"]
            if "auto_update" in updates:
                self.auto_update = bool(updates["auto_update"])
            if "channel" in updates:
                self.update_channel = str(updates["channel"])

        if "logging" in data and isinstance(data["logging"], dict):
            log_sec = data["logging"]
            if "level" in log_sec:
                self.log_level = str(log_sec["level"])
            if "file" in log_sec:
                self.log_file = os.path.expanduser(str(log_sec["file"]))

        return self

    @classmethod
    def _derive_ws_url(cls, api_base_url: str) -> str:
        """Derive standard WebSocket endpoint from API base URL."""
        server = api_base_url.rstrip("/")
        if server.startswith("https://"):
            clean_url = server[len("https://") :]
            return f"wss://{clean_url}/ws/provider/"
        elif server.startswith("http://"):
            clean_url = server[len("http://") :]
            return f"ws://{clean_url}/ws/provider/"
        else:
            return f"wss://{server}/ws/provider/"

    @classmethod
    def from_args(cls, args: Any = None) -> "AgentConfig":
        config = cls()

        # 1. Config file (from args or default paths)
        config_path_str = getattr(args, "config", None) or os.environ.get("NEURALOPS_CONFIG")
        config_candidates = []
        if config_path_str:
            config_candidates.append(Path(config_path_str))
        config_candidates.extend([
            Path("agent-config.yaml"),
            Path.home() / ".neuralops-agent" / "agent-config.yaml",
            Path.home() / ".neuralops-agent" / "config.yaml",
        ])

        for candidate in config_candidates:
            if candidate.is_file():
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                        config.update_from_dict(data)
                    break
                except Exception:
                    pass

        # 2. CLI args override
        if args is not None:
            if getattr(args, "api_key", None):
                config.api_key = args.api_key
            if getattr(args, "server", None):
                config.api_base_url = args.server.rstrip("/")
                config.ws_url = cls._derive_ws_url(config.api_base_url)
            if getattr(args, "ws_url", None):
                config.ws_url = args.ws_url
            if getattr(args, "listing_id", None):
                config.listing_id = args.listing_id
            if getattr(args, "log_level", None):
                config.log_level = args.log_level

        # 3. Env vars override
        if os.environ.get("NEURALOPS_API_KEY"):
            config.api_key = os.environ["NEURALOPS_API_KEY"]
        if os.environ.get("NEURALOPS_SERVER"):
            config.api_base_url = os.environ["NEURALOPS_SERVER"].rstrip("/")
            if not os.environ.get("NEURALOPS_WS_URL"):
                config.ws_url = cls._derive_ws_url(config.api_base_url)
        if os.environ.get("NEURALOPS_WS_URL"):
            config.ws_url = os.environ["NEURALOPS_WS_URL"]
        if os.environ.get("NEURALOPS_LISTING_ID"):
            config.listing_id = os.environ["NEURALOPS_LISTING_ID"]
        if os.environ.get("NEURALOPS_HEARTBEAT_INTERVAL"):
            try:
                config.heartbeat_interval_seconds = int(os.environ["NEURALOPS_HEARTBEAT_INTERVAL"])
            except ValueError:
                pass
        if os.environ.get("NEURALOPS_LOG_LEVEL"):
            config.log_level = os.environ["NEURALOPS_LOG_LEVEL"]
        if os.environ.get("NEURALOPS_DATA_DIR"):
            config.data_dir = os.path.expanduser(os.environ["NEURALOPS_DATA_DIR"])
        if os.environ.get("NEURALOPS_JOB_DIR"):
            config.job_dir = os.path.expanduser(os.environ["NEURALOPS_JOB_DIR"])
        if os.environ.get("NEURALOPS_LOG_DIR"):
            config.log_dir = os.path.expanduser(os.environ["NEURALOPS_LOG_DIR"])
        if os.environ.get("NEURALOPS_AUTO_UPDATE"):
            config.auto_update = os.environ["NEURALOPS_AUTO_UPDATE"].lower() in ("true", "1", "yes")

        # 4. Build and ensure derived paths exist
        os.makedirs(config.data_dir, exist_ok=True)
        os.makedirs(config.job_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        if not config.log_file:
            config.log_file = str(Path(config.log_dir) / "agent.log")

        return config

    def to_dict(self, mask_secrets: bool = False) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        api_k = self.api_key
        if mask_secrets and api_k:
            api_k = f"{api_k[:4]}...{api_k[-4:]}" if len(api_k) > 8 else "***"

        return {
            "api_key": api_k,
            "api_base_url": self.api_base_url,
            "ws_url": self.ws_url,
            "listing_id": self.listing_id,
            "version": self.version,
            "base_price_per_hour": self.base_price_per_hour,
            "listing_type": self.listing_type,
            "location": self.location,
            "storage_gb": self.storage_gb,
            "bandwidth_mbps": self.bandwidth_mbps,
            "supports_ssh": self.supports_ssh,
            "supports_jupyter": self.supports_jupyter,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "gpu_memory_fraction": self.gpu_memory_fraction,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "heartbeat_timeout_seconds": self.heartbeat_timeout_seconds,
            "max_missed_pings": self.max_missed_pings,
            "reconnect_initial_backoff": self.reconnect_initial_backoff,
            "reconnect_max_backoff": self.reconnect_max_backoff,
            "data_dir": self.data_dir,
            "job_dir": self.job_dir,
            "log_dir": self.log_dir,
            "auto_update": self.auto_update,
            "update_channel": self.update_channel,
            "log_level": self.log_level,
            "log_file": self.log_file,
        }
