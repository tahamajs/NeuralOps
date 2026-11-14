"""Deprecation forwarding shim for serving_matrix_verifier."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.serving_matrix_verifier is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.serving_matrix_verifier. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.serving_matrix_verifier import *  # noqa: F403, F401
