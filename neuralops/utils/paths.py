from pathlib import Path


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory (uni/Project)."""
    # Current file is in neuralops/utils/paths.py, project root is 4 levels up
    return Path(__file__).resolve().parent.parent.parent.parent


def get_datasets_dir() -> Path:
    """Returns the absolute path to the global datasets directory."""
    return get_project_root() / "datasets"


def get_glue_dir() -> Path:
    """Returns the absolute path to the local GLUE datasets."""
    return get_datasets_dir() / "glue"


def resolve_path(relative_path: str) -> str:
    """Resolves a path relative to the project root and returns as string."""
    return str((get_project_root() / relative_path).resolve())
