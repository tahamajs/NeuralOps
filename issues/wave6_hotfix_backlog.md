# Wave 6 Hotfix Backlog

This backlog tracks findings, edge cases, minor doc mismatches, and improvements discovered during Wave 5 integration, verification, and acceptance testing.
Per Wave 5 protocol, findings are logged here rather than patched inline to prevent regressions.

| Finding ID | Discovered By | Subsystem | Severity | File / Symbol | Description | Recommended Fix |
|---|---|---|---|---|---|---|

## W5-11: mypy Typing Violations (2026-11-15T16:18:36Z)

| Scope | Error Count |
|---|---|
| mypy --strict (core modules) | 948 |
| mypy full repo (--ignore-missing-imports) | 897 |

### Top Error Categories (strict mode):
- [arg-type]: 252 occurrences
- [operator]: 123 occurrences
- [assignment]: 119 occurrences
- [no-any-return]: 88 occurrences
- [index]: 79 occurrences
- [union-attr]: 76 occurrences
- [no-untyped-def]: 37 occurrences
- [type-arg]: 34 occurrences
- [no-untyped-call]: 33 occurrences
- [return-value]: 17 occurrences

**Remediation:** These must be fixed before v2.5.0 strict typing gate.
