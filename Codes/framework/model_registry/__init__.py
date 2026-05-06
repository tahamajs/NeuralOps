# framework/model_registry/__init__.py
from .registry import ModelRegistry, ModelEntry

# framework/model_registry/registry.py
class ModelEntry:
    def __init__(self, name, version, artifact_path, metrics, tags, framework="pytorch"):
        self.name = name
        self.version = version
        self.artifact_path = artifact_path
        self.metrics = metrics
        self.tags = tags
        self.framework = framework

class ModelRegistry:
    entries = []

    @classmethod
    def register(cls, artifact_path, metrics, tags, name=None, version=None):
        # Creates a ModelVersion entry in Django + links to registry
        entry = ModelEntry(name=name or "model", version=version or "v1",
                           artifact_path=artifact_path, metrics=metrics, tags=tags)
        cls.entries.append(entry)
        # Also call Django's ModelVersionService.create_version(...)
        return entry

    @classmethod
    def deploy(cls, entry, target="torchserve", **kwargs):
        if target == "torchserve":
            from framework.deployment.torchserve import deploy_to_torchserve
            return deploy_to_torchserve(entry, **kwargs)
        elif target == "onnx":
            from framework.deployment.onnx import export_to_onnx
            return export_to_onnx(entry, **kwargs)
        elif target == "hf_hub":
            from framework.deployment.hf_hub import push_to_hf_hub
            return push_to_hf_hub(entry, **kwargs)