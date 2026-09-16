"""Apply variables.json to all .md files."""

import argparse
import json
import re
from pathlib import Path

PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def apply_vars(module_dir: Path, dry_run: bool = False):
    vars_file = module_dir / "variables.json"
    if not vars_file.exists():
        print(f"Not found: {vars_file}")
        return

    variables = json.loads(vars_file.read_text(encoding="utf-8"))
    variables = {k: v for k, v in variables.items() if not k.startswith("_")}

    print(f"Loaded {len(variables)} variables\n")

    for md_file in sorted(module_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        missing = set()

        def replace(match):
            key = match.group(1)
            if key in variables:
                return str(variables[key])
            missing.add(key)
            return match.group(0)

        new = PATTERN.sub(replace, content)

        if new != content:
            if dry_run:
                print(f"[DRY] {md_file.name}")
            else:
                md_file.write_text(new, encoding="utf-8")
                print(f"Updated {md_file.name}")

        if missing:
            print(f"  Missing: {sorted(missing)}")

    print("\nDone." if not dry_run else "\nDry run.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    md = Path(__file__).parent / args.module
    if not md.exists():
        print(f"Not found: {md}")
        return

    apply_vars(md, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
```

---