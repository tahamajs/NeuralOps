"""Central Prompt Registry – CRUD, versioning, aliases, search."""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .config import PromptConfig, PromptType, StorageBackend
from .templates import PromptTemplate, ChatPromptTemplate
from .versioning import PromptVersion, VersionHistory
from .evaluation import PromptEvaluation

logger = logging.getLogger(__name__)

class PromptRegistry:
    """
    Centralised prompt registry with Git‑like versioning, aliases,
    and pluggable storage backends.

    Modeled after: MLflow Prompt Registry, LangSmith Prompt Hub, Qwak Prompts.

    Usage:
        registry = PromptRegistry(PromptConfig(registry_path="./prompts"))

        # Register a new prompt (creates version 1)
        registry.register("customer_support",
                          PromptTemplate("You are {{role}}. Answer: {{question}}"),
                          commit_message="Initial customer support prompt")

        # Create a new version
        registry.update("customer_support",
                        PromptTemplate("You are a helpful {{role}}. {{question}}"),
                        commit_message="Improved tone")

        # Set production alias
        registry.set_alias("customer_support", "production", version=2)

        # Load the production version
        prompt = registry.load("customer_support", alias="production")
        result = prompt.render(role="assistant", question="Help!")
    """

    def __init__(self, config: PromptConfig = None):
        self.config = config or PromptConfig()
        self._prompts: Dict[str, VersionHistory] = {}   # name → version chain
        self._aliases: Dict[str, Dict[str, int]] = {}    # name → {alias → version}
        self._evaluations: Dict[str, Dict[int, PromptEvaluation]] = {}
        self._load_from_disk()

    # ── Registration ────────────────────────────────────────
    def register(self, name: str,
                 template: Union[PromptTemplate, ChatPromptTemplate],
                 commit_message: str = "",
                 author: str = "system",
                 tags: List[str] = None,
                 metadata: Dict[str, Any] = None) -> PromptVersion:
        """Create a new prompt (version 1)."""
        if name in self._prompts:
            raise ValueError(f"Prompt '{name}' already exists. Use update() to add a version.")
        v = self._create_version(name, template, commit_message, author, tags, metadata)
        self._get_history(name).add(v)
        self._set_default_alias(name, v.version_number)
        self._save(name)
        logger.info("Registered prompt '%s' – v%d (%s)", name, v.version_number, v.id[:8])
        return v

    def update(self, name: str,
               template: Union[PromptTemplate, ChatPromptTemplate],
               commit_message: str = "",
               author: str = "system",
               tags: List[str] = None,
               metadata: Dict[str, Any] = None) -> PromptVersion:
        """Add a new version to an existing prompt."""
        if name not in self._prompts:
            raise KeyError(f"Prompt '{name}' not found. Use register() first.")
        v = self._create_version(name, template, commit_message, author, tags, metadata)
        self._get_history(name).add(v)
        self._save(name)
        logger.info("Updated prompt '%s' – v%d (%s)", name, v.version_number, v.id[:8])
        return v

    def _create_version(self, name, template, commit_message, author, tags, metadata):
        raw = template.template if isinstance(template, PromptTemplate) else json.dumps(template.messages)
        ttype = "simple" if isinstance(template, PromptTemplate) else "chat"
        ts = datetime.now().isoformat()
        vid = PromptVersion.compute_id(name, raw, ts)
        return PromptVersion(
            id=vid, version_number=0, prompt_name=name,
            template=raw, template_type=ttype,
            commit_message=commit_message, author=author,
            tags=tags or [], metadata=metadata or {}
        )

    # ── Loading ────────────────────────────────────────────
    def load(self, name: str, version: int = None,
             alias: str = None) -> Union[PromptTemplate, ChatPromptTemplate]:
        """Retrieve a prompt by version number or alias.
           If neither is given, returns the latest version.
        """
        history = self._prompts.get(name)
        if history is None:
            raise KeyError(f"Prompt '{name}' not found.")
        if alias:
            vnum = self._aliases.get(name, {}).get(alias)
            if vnum is None:
                raise KeyError(f"Alias '{alias}' not found for prompt '{name}'.")
            v = history.get(vnum)
        elif version:
            v = history.get(version)
        else:
            v = history.latest()
        if v is None:
            raise ValueError(f"Version not found for prompt '{name}'.")
        return self._deserialise(v)

    def load_production(self, name: str):
        """Shortcut to load the production alias."""
        return self.load(name, alias="production")

    def _deserialise(self, v: PromptVersion):
        if v.template_type == "chat":
            return ChatPromptTemplate(messages=json.loads(v.template))
        return PromptTemplate(template=v.template, metadata=v.metadata)

    # ── Aliases / Environments ─────────────────────────────
    def set_alias(self, name: str, alias: str, version: int) -> None:
        """Point an alias (e.g. 'production', 'staging') to a specific version."""
        if name not in self._prompts:
            raise KeyError(f"Prompt '{name}' not found.")
        if alias not in self._aliases.get(name, {}):
            self._aliases.setdefault(name, {})["_history"] = []
        self._aliases.setdefault(name, {})["_history"].append(
            {"alias": alias, "version": version, "timestamp": datetime.now().isoformat()}
        )
        self._aliases[name][alias] = version
        self._save(name)
        logger.info("Alias '%s' for '%s' → v%d", alias, name, version)

    def get_alias(self, name: str, alias: str) -> Optional[int]:
        return self._aliases.get(name, {}).get(alias)

    def alias_history(self, name: str, alias: str) -> List[dict]:
        """Return the full assignment history for an alias."""
        return [h for h in self._aliases.get(name, {}).get("_history", [])
                if h["alias"] == alias]

    def rollback_alias(self, name: str, alias: str):
        """Roll back an alias to its previous version."""
        history = self.alias_history(name, alias)
        if len(history) < 2:
            raise ValueError(f"No previous version to roll back for alias '{alias}'.")
        prev = history[-2]["version"]
        self.set_alias(name, alias, prev)
        logger.info("Rolled back alias '%s' for '%s' → v%d", alias, name, prev)

    # ── Search & List ──────────────────────────────────────
    def list_prompts(self) -> List[Dict[str, Any]]:
        return [
            {"name": n, "versions": len(h.versions),
             "latest": h.latest().version_number if h.latest() else 0,
             "aliases": self._aliases.get(n, {}).get("_aliases", {}),
             "tags": list(set(t for v in h.versions for t in v.tags))}
            for n, h in self._prompts.items()
        ]

    def search(self, tag: str = None, query: str = None) -> List[Dict[str, Any]]:
        results = self.list_prompts()
        if tag:
            results = [r for r in results if tag in r.get("tags", [])]
        if query:
            q = query.lower()
            results = [r for r in results if q in r["name"].lower()]
        return results

    # ── Version History ────────────────────────────────────
    def get_history(self, name: str) -> VersionHistory:
        return self._prompts.get(name)

    def diff(self, name: str, v1: int, v2: int = None) -> str:
        h = self._prompts.get(name)
        if h is None: raise KeyError(f"Prompt '{name}' not found.")
        return h.diff(v1, v2)

    # ── Evaluation (delegates to PromptEvaluation) ─────────
    def log_evaluation(self, name: str, version: int, **metrics):
        if name not in self._evaluations:
            self._evaluations[name] = {}
        self._evaluations[name][version] = PromptEvaluation(
            prompt_name=name, version=version, **metrics
        )
        self._save(name)

    def get_evaluations(self, name: str) -> Dict[int, "PromptEvaluation"]:
        return self._evaluations.get(name, {})

    # ── Persistence ─────────────────────────────────────────
    def _get_history(self, name: str) -> VersionHistory:
        if name not in self._prompts:
            self._prompts[name] = VersionHistory()
        return self._prompts[name]

    def _set_default_alias(self, name, version):
        self._aliases.setdefault(name, {"_history": []})
        self._aliases[name][self.config.default_alias] = version

    def _save(self, name: str):
        if self.config.storage_backend != StorageBackend.LOCAL:
            return
        os.makedirs(self.config.registry_path, exist_ok=True)
        data = {
            "versions": self._prompts[name].to_list(),
            "aliases": {k: v for k, v in self._aliases.get(name, {}).items() if k != "_history"},
            "alias_history": self._aliases.get(name, {}).get("_history", []),
            "evaluations": {str(k): v.to_dict() for k, v in
                            self._evaluations.get(name, {}).items()},
        }
        path = os.path.join(self.config.registry_path, f"{name}.json")
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _load_from_disk(self):
        if self.config.storage_backend != StorageBackend.LOCAL:
            return
        p = self.config.registry_path
        if not os.path.isdir(p):
            return
        for fname in os.listdir(p):
            if not fname.endswith('.json'):
                continue
            name = fname[:-5]
            with open(os.path.join(p, fname)) as f:
                data = json.load(f)
            history = VersionHistory()
            for vd in data.get("versions", []):
                history.add(PromptVersion(**vd))
            self._prompts[name] = history
            aliases = data.get("aliases", {})
            aliases["_history"] = data.get("alias_history", [])
            self._aliases[name] = aliases
            evals = {}
            for k, v in data.get("evaluations", {}).items():
                evals[int(k)] = PromptEvaluation(**v)
            self._evaluations[name] = evals