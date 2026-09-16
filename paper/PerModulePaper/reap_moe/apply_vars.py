"""Apply variables.json to all .md files."""
import argparse, json, re
from pathlib import Path

PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def apply(md: Path, dry: bool = False):
    vf = md / "variables.json"
    if not vf.exists():
        print(f"Not found: {vf}")
        return

    vars = json.loads(vf.read_text(encoding="utf-8"))
    vars = {k: v for k, v in vars.items() if not k.startswith("_")}
    print(f"Loaded {len(vars)} vars\n")

    for f in sorted(md.glob("*.md")):
        c = f.read_text(encoding="utf-8")
        missing = set()

        def replace(m):
            k = m.group(1)
            if k in vars:
                return str(vars[k])
            missing.add(k)
            return m.group(0)

        new = PATTERN.sub(replace, c)
        if new != c:
            if dry:
                print(f"[DRY] {f.name}")
            else:
                f.write_text(new, encoding="utf-8")
                print(f"Updated {f.name}")
        if missing:
            print(f"  Missing: {sorted(missing)}")

    print("\nDone." if not dry else "\nDry run.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--module", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    md = Path(__file__).parent / args.module
    if not md.exists():
        print(f"Not found: {md}")
        return
    apply(md, args.dry_run)


if __name__ == "__main__":
    main()
```

---