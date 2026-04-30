"""
SVD‑Lin Studio – Complete Framework for Transformer Compression & Analysis
"""

__version__ = "2.0.0"
__author__ = "SVD-Lin Team"

from . import (
    config,
    data,
    compression,
    attentions,
    routing,
    ensemble,
    training,
    analysis,
    visualization,
    soa,
    evaluation,
    utils,
)

from .cli import main