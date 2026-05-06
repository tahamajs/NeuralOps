
# framework/prompts/tracker.py
import json, hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

@dataclass
class PromptVersion:
    id: str; template: str; variables: Dict[str, Any]
    created_at: str; parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class PromptTracker:
    def __init__(self, storage_path: str = "./prompts"):
        self.storage_path = storage_path
        self._versions: List[PromptVersion] = []
        import os; os.makedirs(storage_path, exist_ok=True)

    def register(self, template: str, variables: Dict[str, Any] = None,
                 parent_id: str = None, metadata: Dict = None) -> PromptVersion:
        vid = hashlib.sha256(f"{template}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        v = PromptVersion(id=vid, template=template, variables=variables or {},
                          created_at=datetime.now().isoformat(), parent_id=parent_id,
                          metadata=metadata or {})
        self._versions.append(v)
        self._save(v)
        return v

    def render(self, version_id: str, values: Dict[str, Any]) -> str:
        v = next((x for x in self._versions if x.id == version_id), None)
        if v is None: raise KeyError(f"Version {version_id} not found")
        all_vars = {**v.variables, **values}
        return v.template.format(**all_vars)

    def compare(self, v1: str, v2: str) -> Dict[str, Any]:
        a = next((x for x in self._versions if x.id == v1), None)
        b = next((x for x in self._versions if x.id == v2), None)
        return {"template_changed": a.template != b.template if a and b else None,
                "variables_added": list(set(b.variables) - set(a.variables)) if a and b else [],
                "variables_removed": list(set(a.variables) - set(b.variables)) if a and b else []}

    def _save(self, v: PromptVersion):
        with open(f"{self.storage_path}/{v.id}.json", 'w') as f:
            json.dump(v.__dict__, f, indent=2, default=str)