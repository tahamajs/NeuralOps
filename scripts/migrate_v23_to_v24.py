"""Automated migration script for upgrading user codebases from NeuralOps v2.3 to v2.4.0."""

import re
import sys


def migrate_file(path: str) -> None:
    """Migrate a single file from v2.3 to v2.4 patterns."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Migrate serving method
    content = content.replace(
        "continuous_batching_generate",
        "generate_sequential",
    )

    # Migrate deprecated AMP imports
    content = content.replace(
        "from torch.cuda.amp import autocast, GradScaler",
        "from torch.amp import autocast, GradScaler",
    )

    # Migrate YAML loaders
    content = re.sub(
        r'yaml\.safe_load\(open\(["\']([^"\']+)["\']\)\)',
        r'load_benchmark_config("\1")',
        content,
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Migrated: {path}")


def main() -> None:
    """CLI entrypoint for batch file migration."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/migrate_v23_to_v24.py <file1.py> [file2.py ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        migrate_file(path)


if __name__ == "__main__":
    main()
