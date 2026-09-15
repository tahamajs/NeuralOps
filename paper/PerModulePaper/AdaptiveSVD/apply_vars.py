"""Apply variables.json to all .md files."""

import argparse
import json
import re
from pathlib import Path

PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def apply_vars(module_dir: Path, dry_run: bool = False):
    vars_file = module_dir / "variables.json"
    if not vars_file.exists():
        print(f"variables.json not found in {module_dir}")
        return
    
    variables = json.loads(vars_file.read_text(encoding="utf-8"))
    variables = {k: v for k, v in variables.items() if not k.startswith("_")}
    
    print(f"Loaded {len(variables)} variables")
    print()
    
    for md_file in sorted(module_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        missing = set()
        
        def replace(match):
            key = match.group(1)
            if key in variables:
                return str(variables[key])
            missing.add(key)
            return match.group(0)
        
        new_content = PATTERN.sub(replace, content)
        
        if new_content != content:
            if dry_run:
                print(f"[DRY] Would update {md_file.name}")
            else:
                md_file.write_text(new_content, encoding="utf-8")
                print(f"Updated {md_file.name}")
        
        if missing:
            print(f"  Missing in {md_file.name}: {sorted(missing)}")
    
    print()
    print("Dry run complete." if dry_run else "Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    module_dir = Path(__file__).parent / args.module
    if not module_dir.exists():
        print(f"Not found: {module_dir}")
        return
    
    apply_vars(module_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
