import json
import os
from typing import Any, Dict


class LocalBackend:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)

    def save(self, name: str, data: Dict[str, Any]):
        with open(os.path.join(self.path, f"{name}.json"), "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, name: str) -> Dict[str, Any]:
        with open(os.path.join(self.path, f"{name}.json")) as f:
            return json.load(f)

    def list_names(self) -> list:
        return [f[:-5] for f in os.listdir(self.path) if f.endswith(".json")]
