"""
Model Versioning service.

Provides a lazy‑loading interface to the Django `ModelVersion` model,
so the service can be imported even before Django apps are loaded.
"""

from .services import ModelVersionService

__all__ = ["ModelVersionService"]

# Convenience alias
create_version = ModelVersionService.create_version
load_version = ModelVersionService.load_version
delete_old_versions = ModelVersionService.delete_old_versions