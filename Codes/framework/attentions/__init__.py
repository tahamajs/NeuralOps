"""Automatically import sub‑folders to trigger registration."""
from . import core
try:
    from . import hybrid_ssm
except ImportError:
    pass
# Add future modules here as they are created
