import os
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

from datasets import load_dataset
tasks = [
    ("piqa", "https://huggingface.co/datasets/ybisk/piqa/resolve/refs%2Fconvert%2Fparquet/plain_text/validation/0000.parquet"),
    ("arc_easy", "https://huggingface.co/datasets/allenai/ai2_arc/resolve/refs%2Fconvert%2Fparquet/ARC-Easy/test/0000.parquet"),
    ("arc_challenge", "https://huggingface.co/datasets/allenai/ai2_arc/resolve/refs%2Fconvert%2Fparquet/ARC-Challenge/test/0000.parquet"),
    ("hellaswag", "https://huggingface.co/datasets/Rowan/hellaswag/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"),
    ("winogrande", "https://huggingface.co/datasets/allenai/winogrande/resolve/refs%2Fconvert%2Fparquet/winogrande_xl/validation/0000.parquet"),
]

for name, url in tasks:
    try:
        ds = load_dataset("parquet", data_files=url, split="train")
        print(f"{name}: OK ({len(ds)} samples)")
    except Exception as e:
        print(f"{name}: FAILED ({type(e).__name__}: {e})")
