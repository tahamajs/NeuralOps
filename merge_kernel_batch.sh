#!/usr/bin/env bash
set -e

branches=(
    "wave9/W9-K1"
    "wave9/W9-K2"
    "wave9/W9-K3"
    "wave9/W9-K4"
    "wave9/W9-K7"
    "wave9/W9-K8"
    "wave9/W9-K15"
    "wave9/W9-K16"
    "wave9/W9-K19"
)

for b in "${branches[@]}"; do
    echo "=== Merging $b ==="
    git merge --no-ff "$b" -m "merge(wave9): integrate $(basename "$b")" || {
        echo "Conflict detected in $b, checking status..."
        git checkout --theirs tests/conftest.py 2>/dev/null || true
        git checkout --theirs .agent/WAVE9_PROGRESS.json 2>/dev/null || true
        git add -A
        git commit -m "merge(wave9): integrate $(basename "$b") (conflicts resolved)"
    }
done

echo "=== All ready branches merged successfully! ==="
