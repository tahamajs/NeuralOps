"""
Utility to scan and list available datasets.
"""

import os


def list_local_datasets(base_dir: str) -> list[str]:
    """Return a list of dataset names found in the given directory."""
    if not os.path.isdir(base_dir):
        return []
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
