def upload(local_path, **kwargs) -> None:
    """Upload."""
    pass  # already on disk


def download(local_path, **kwargs) -> None:
    """Download."""
    pass  # assumed to exist


def delete(local_path, **kwargs) -> None:
    """Delete."""
    import os
    import shutil

    if os.path.exists(local_path):
        shutil.rmtree(local_path, ignore_errors=True)
