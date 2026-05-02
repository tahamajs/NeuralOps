import logging
from .registry import register_method

logger = logging.getLogger(__name__)

def get_llrd_param_groups(model, base_lr, decay, weight_decay):
    # ... (legacy code)

@register_method("llrd")
def apply_llrd(model, config):
    """LLRD does not modify model; just returns param groups. The trainer will use them."""
    # We return a dummy function so that `main.py` knows to use LLRD optimizer
    pass