# Repository Tag & Release Policy

This document establishes the release and tagging conventions for `tahamajs/SVD_linear_Attention`.

---

## 1. Tag Naming Conventions

### Allowed Tag Patterns

1. **Semantic Versioning Releases**:
   - Format: `v<MAJOR>.<MINOR>.<PATCH>` (e.g., `v1.0.0`, `v1.1.0`, `v1.2.0`)
   - Annotated tags only: `git tag -a vX.Y.Z -m "Release description"`
   - Each semver tag should correspond to an official release published or staged for PyPI.

2. **Pre-releases**:
   - Format: `v<MAJOR>.<MINOR>.<PATCH>-(alpha|beta|rc).<N>` (e.g., `v1.3.0-rc.1`)

3. **Key Scientific / Experimental Milestones**:
   - Format: `<experiment-name>-complete-v<N>` or `milestone-<name>` (e.g., `kd-experiment-complete-v1`)
   - Reserved strictly for key paper-reproducible checkpoints.

---

## 2. Prohibited Tag Patterns

Do **NOT** push tags matching any of the following:
- Temporary backups: `backup-*`, `backup_*`
- Pre-refactor checkpoints: `pre-refactor-*`, `pre-audit-*`
- Timestamped tags: `*_[0-9]{8}_[0-9]{4}`, `*-[0-9]{8}`
- WIP / Scratch tags: `wip-*`, `temp-*`, `test-*`

*Use temporary Git branches (e.g. `feature/...` or `tmp/...`) rather than tags for in-progress checkpoints.*

---

## 3. Creating a Release Tag

```bash
git tag -a v1.3.0 -m "Release v1.3.0: summary of changes"
git push origin v1.3.0
```

---

## 4. Deleting Errant Tags

If an errant or temporary tag is pushed by mistake:

```bash
# Delete locally
git tag -d <tag_name>

# Delete from remote
git push --delete origin <tag_name>

# Synchronize across development servers
git fetch --prune --prune-tags origin
```
