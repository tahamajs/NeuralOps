# Wave 6 Hotfix Backlog

This backlog tracks findings, edge cases, minor doc mismatches, and improvements discovered during Wave 5 integration, verification, and acceptance testing.
Per Wave 5 protocol, findings are logged here rather than patched inline to prevent regressions.

| Finding ID | Discovered By | Subsystem | Severity | File / Symbol | Description | Recommended Fix |
|---|---|---|---|---|---|---|
| W5-2-01 | W5-2 | worktrees / git | Low (Maintenance) | `/Users/tahamajs/dev/Project_W4_*` | 20 Wave 4 git worktrees remain active in git worktree list despite being 100% merged into main. | Execute git worktree remove / git worktree prune for Wave 4 worktrees post-v2.4.0 release. |
| W5-2-02 | W5-2 | documentation / orchestration | Low (Documentation) | `.agent/ORCHESTRATOR_WAVE3.md` | Legacy Wave 3 orchestrator lacks an explicit banner marking it superseded by Wave 4. | Add an archival banner referencing .agent/WAVE4_EXECUTE.md and audits/wave4_master.md. |

