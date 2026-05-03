from . import minimal_aurora
from . import aurora

# Optionally make the injection functions available at top level
from .minimal_aurora import MinimalAURORAAttention, inject_minimal_aurora
from .aurora import AURORAAttention, inject_aurora