from dataclasses import dataclass


@dataclass
class HardwareConfig:
    device: str = "auto"  # auto, cuda, mps, cpu
    force_cpu_for_svd: bool = True
    enable_fp16: bool = False
    enable_bf16: bool = False
    num_workers: int = 4
    pin_memory: bool = False
    torch_compile: bool = False
    empty_cache_every_step: bool = False
