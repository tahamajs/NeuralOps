# 5. Architecture and Implementation
The module integrates directly into standard transformer attention and feed-forward sublayers via `replace_module`. Projections are converted to `SVDLinear` instances with frozen base parameters and optional LoRA adapters.
