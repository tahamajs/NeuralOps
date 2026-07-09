"""Agent configuration from YAML file + CLI args + env vars."""
import os, yaml
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class AgentConfig:
    api_key: str = ''
    api_base_url: str = 'https://api.neuralops.com'
    ws_url: str = 'wss://api.neuralops.com/ws/provider/'
    listing_id: str = ''
    version: str = '1.0.0'

    # Provider settings
    base_price_per_hour: float = 0.50
    listing_type: str = 'on_demand'  # on_demand | spot | reserved
    location: str = ''
    storage_gb: int = 100
    supports_ssh: bool = True
    supports_jupyter: bool = False

    # Resource limits
    max_concurrent_jobs: int = 2
    gpu_memory_fraction: float = 0.90  # leave 10% for system

    # Network
    speed_test_interval_hours: int = 24
    heartbeat_interval_seconds: int = 30

    # Paths
    data_dir: str = str(Path.home() / '.neuralops-agent')
    job_dir: str = str(Path.home() / '.neuralops-agent' / 'jobs')
    log_dir: str = str(Path.home() / '.neuralops-agent' / 'logs')

    # Auto-update
    auto_update: bool = True
    update_channel: str = 'stable'

    # Logging
    log_level: str = 'INFO'
    log_file: str = ''

    @classmethod
    def from_args(cls, args) -> 'AgentConfig':
        config = cls()
        # 1. Default config file
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
                for k, v in data.items():
                    if hasattr(config, k):
                        setattr(config, k, v)
        # 2. CLI args override
        if args.api_key:
            config.api_key = args.api_key
        if args.server:
            config.api_base_url = args.server
            ws_proto = 'wss://' if args.server.startswith('https://') else 'ws://'
            clean_url = args.server.replace('https://', '').replace('http://', '')
            config.ws_url = f"{ws_proto}{clean_url}/ws/provider/"
        # 3. Env vars override
        if os.environ.get('NEURALOPS_API_KEY'):
            config.api_key = os.environ['NEURALOPS_API_KEY']
        if os.environ.get('NEURALOPS_SERVER'):
            config.api_base_url = os.environ['NEURALOPS_SERVER']
            server_url = os.environ['NEURALOPS_SERVER']
            ws_proto = 'wss://' if server_url.startswith('https://') else 'ws://'
            clean_url = server_url.replace('https://', '').replace('http://', '')
            config.ws_url = f"{ws_proto}{clean_url}/ws/provider/"
        # Build derived paths
        os.makedirs(config.data_dir, exist_ok=True)
        os.makedirs(config.job_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        config.log_file = str(Path(config.log_dir) / 'agent.log')
        return config