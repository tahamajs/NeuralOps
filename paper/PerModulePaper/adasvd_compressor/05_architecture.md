# 5. Architecture and Implementation
Dense layers are replaced with `SVDLinear` modules ($y = (x B^	op) A^	op + b$). Supports Conv1D transposition for GPT-2 and preservation of LoRA adapters.
