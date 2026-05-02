# framework/__init__.py
"""Efficient Attention Framework – minimal bootstrap."""

# These imports are safe because they use the new flat structure
from . import config
from . import data
from . import trainer
from . import utils

# -------------------------------------------------------------------
# This line triggers the attention modules' registration.
# It runs framework/attentions/__init__.py, which imports 'nexus'
# (and any other modules you add) → their @ATTENTION_REGISTRY.register()
# decorators execute, making them available in ATTENTION_REGISTRY.
# -------------------------------------------------------------------
from . import attentions