"""Security configuration – currently only holds the default algorithm."""
from typing import Optional

from dataclasses import dataclass


@dataclass
class SecurityConfig:
    """Optional security settings.

    Attributes:
        algorithm: signature algorithm. Currently only 'ed25519' is supported.
        enforce_weights_only: whether to strictly enforce weights_only=True during checkpoint loading.
        prefer_safetensors: whether to prefer safetensors format when loading/saving weights.
        allowed_extensions: permissible model checkpoint extensions.
    """

    algorithm: str = "ed25519"
    enforce_weights_only: bool = True  # FIX: Enforce weights_only=True by default for secure loading
    prefer_safetensors: bool = True    # FIX: Prefer safetensors format to prevent arbitrary code execution
    allowed_extensions: tuple = (".safetensors", ".sft", ".pt", ".pth", ".bin", ".ckpt")

