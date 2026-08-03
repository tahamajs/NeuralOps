"""Security configuration – currently only holds the default algorithm."""
from typing import Optional

from dataclasses import dataclass


@dataclass
class SecurityConfig:
    """Optional security settings.

    Attributes:
        algorithm: signature algorithm. Currently only 'ed25519' is supported.
    """

    algorithm: str = "ed25519"
