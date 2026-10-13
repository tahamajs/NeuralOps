"""Backward-compatibility shim. Moved to neuralops.core.utils.rope."""
import warnings
warnings.warn(
    "neuralops.core.rope is deprecated. Use neuralops.core.utils.rope.",
    DeprecationWarning,
    stacklevel=2,
)
from neuralops.core.utils.rope import *  # noqa: F401,F403
from neuralops.core.utils.rope import (  # noqa: F401
    _rotate_half,
    apply_rotary_pos_emb,
)
