# framework/attentions/__init__.py
"""Import all attention modules to trigger their registration."""
from . import core          # registry + BaseAttention
from . import nexus         # NEXUSAttention registration
# Add future modules here, e.g.:
# from . import leka_v2
# from . import spectra_v2
