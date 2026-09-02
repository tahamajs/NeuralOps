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


def entries_for_paths(paths: list[Path]) -> list[dict[str, str]]:
    """Create catalog entries for files that live across a documentation area."""
    return [
        {"id": path.name, "title": title_for(path), "path": path.relative_to(DOCS).as_posix()}
        for path in sorted(paths, key=lambda p: p.name.lower())
    ]


def main() -> None:
    catalog = {
        "guides": entries_for_paths([
            path for path in DOCS.glob("*.md")
            if path.name not in {"README.md"}
            and "verification" not in path.stem
        ]),
        "verification": entries_for_paths(
            list(DOCS.glob("*_verification.md"))
            + list(CONTENT.glob("*_validation_report.md"))
        ),
        "reports": entries(DOCS / "reports", "reports") + entries(DOCS / "results", "results")
        + entries_for_paths(list(CONTENT.glob("*benchmark_report.md"))),
        "studios": entries(DOCS / "Attention_Visualization_Dashboard", "Attention_Visualization_Dashboard")
        + entries(
            DOCS / "Mapping_Apply_the_Attention_Style_Interactive_Dashboard_to_ALL_Transformer_Components",
            "Mapping_Apply_the_Attention_Style_Interactive_Dashboard_to_ALL_Transformer_Components",
        ),
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
