import os
from transformers import AutoConfig

models_dir = 'models'
for d in os.listdir(models_dir):
    p = os.path.join(models_dir, d)
    if os.path.isdir(p):
        try:
            c = AutoConfig.from_pretrained(p)
            max_len = getattr(c, 'max_position_embeddings', getattr(c, 'n_positions', getattr(c, 'max_sequence_length', None)))
            print(f"{d}: max_len={max_len}")
        except Exception:
            pass
