import sys
from pathlib import Path

# Add project & framework paths
proj_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(proj_dir))
sys.path.insert(0, str(proj_dir / "Codes"))

from neuralops.data.cached_tensors import get_cached_tokenized_dataset

if __name__ == "__main__":
    cache_dir = proj_dir / "data_cache"
    print(f"Preparing tokenized cache in {cache_dir}...")
    ds = get_cached_tokenized_dataset("sst2", "distilbert", 128, cache_dir=cache_dir)
    print(
        f"Successfully generated cache: {len(ds['train'])} train samples, {len(ds['val'])} validation samples."
    )
