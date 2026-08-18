"""Model signing, verification, and integrity checks."""

from .config import SecurityConfig
from .deserialization import (
    is_safe_checkpoint_format,
    safe_load_checkpoint,
    safe_load_model_weights,
    safe_save_checkpoint,
)
from .hashes import compute_file_hash, verify_file_hash
from .signing import generate_keypair, sign_model, verify_signature

__all__ = [
    "sign_model",
    "verify_signature",
    "generate_keypair",
    "compute_file_hash",
    "verify_file_hash",
    "SecurityConfig",
    "safe_load_checkpoint",
    "safe_save_checkpoint",
    "safe_load_model_weights",
    "is_safe_checkpoint_format",
]

