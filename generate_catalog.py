"""Generate the deterministic catalog consumed by the Docs GitHub Pages UI."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "Docs"
CONTENT = DOCS / "content"


def title_for(path: Path) -> str:
    """Prefer the first Markdown H1, with a readable filename fallback."""
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^#\s+(.+?)\s*$", text, flags=re.MULTILINE)
    if match:
        title = re.sub(r"[*_`]+", "", match.group(1)).strip()
        if title:
            return title
    return path.stem.replace("_", " ").replace("-", " ").title()


def entries(directory: Path, public_prefix: str) -> list[dict[str, str]]:
    if not directory.exists():
        return []
    return [
        {
            "id": path.name,
            "title": title_for(path),
            "path": f"{public_prefix}/{path.name}",
        }
        for path in sorted(directory.glob("*.md"), key=lambda p: p.name.lower())
    ]


def main() -> None:
    catalog = {
        "attentions": entries(CONTENT / "attentions", "content/attentions"),
        "compressions": entries(CONTENT / "compressions", "content/compressions"),
        "attentions_blog": entries(CONTENT / "attentions_blog", "content/attentions_blog"),
        "compressions_blog": entries(CONTENT / "compressions_blog", "content/compressions_blog"),
        "blog_series": entries(DOCS / "blog", "blog"),
        "framework": entries(CONTENT / "framework", "content/framework"),
    }
    target = DOCS / "catalog.json"
    target.write_text(json.dumps(catalog, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {sum(len(items) for items in catalog.values())} entries to {target}")


if __name__ == "__main__":
    main()
