import torch.nn as nn, torch.nn.functional as F, logging
from .registry import register_method

logger = logging.getLogger(__name__)

class MixoutLinear(nn.Module):
    # ... (legacy code)

def replace_linears_with_mixout(model, p, skip_classifier=True):
    # ... (recursive replacement)

@register_method("mixout")
def apply_mixout(model, config):
    replace_linears_with_mixout(model, config.mixout_p)