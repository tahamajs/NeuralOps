## WEB-02 — Neutralise the dangerous endpoints: unauthenticated RCE surfaces — Phase P0 — Findings W-C2, W-C3, W-C5, W-C6, W-C7, W-C8, W-C22, W-H46, W-H28, W-H12, W-M22, W-H13

**Depends on:** WEB-01 (migrations exist so the fixed views can be tested)
**Goal:** No unauthenticated remote-code-execution or free-money surface remains: terminal WS, LaTeX compile, raw SQL, Docker exec, artifact traversal, escrow deposit, paper collab WS, agent WS, webhook SSRF, mlops SSRF, git push — all closed, staff-gated, or removed.
**Context (Persian, 2-4 lines):** خطرناک‌ترین یافته‌های ممیزی این‌ها هستند: شل بدون احراز هویت، کامپایل LaTeX دلخواه، SQL خام، اجرای دستور در Docker و واریز مجانی escrow. این پرامپت همه را می‌بندد یا غیرفعال می‌کند — نه «بعداً».
**Prompt to paste (English):** ```
You are working in /Users/tahamajs/Documents/uni/Project_website. Only change
the files listed. Do not touch anything else. Use ONLY the existing project
venv — never create a new venv. Do not commit anything.

## Role + scope
You are a security engineer. Your scope is `django_api/terminal/
consumers.py,routing.py`, `django_api/paper/views.py,compiler.py,consumers.py`,
`django_api/live_dashboard/views.py`, `django_api/integrations/views/
sandbox.py`, `django_api/integrations/models/sandbox.py`,
`django_api/integrations/serializers.py` (sandbox fields only),
`django_api/experiment/views.py` (artifact download only),
`django_api/gpu_escrow/views.py,models.py`, `django_api/gpu_marketplace/
consumers.py`, `django_api/notes/webhook_utils.py,views.py` (webhook test
only), `django_api/mlops/views.py`, `django_api/paper/views.py` (git
push/pull only), `django_api/gpu_validator/views.py` (stake check only).
Run your own multi-sub-agents in parallel: (a) WS surfaces (terminal,
paper collab, agent provider); (b) code-execution surfaces (LaTeX, Docker,
raw SQL, git); (c) money/SSRF surfaces (escrow, webhooks, mlops, validator
stake). You synthesize their reports into ONE final report.

## Background
Exact problems (audit 2026-11-07):
- W-C2 terminal/consumers.py:31-105: connect() accepts first, then
  pty.fork() + os.execvp("/bin/bash"); no staff check; token from query
  string; no idle timeout; one shared "terminal" channel group (cross-user
  leak, proven by terminal/tests.py:36-42).
- W-C3 paper/views.py:851-935: arbitrary `source` → pdflatex with NO
  -no-shell-escape; hardcoded fake success log (:920-926); `load_paper_source`
  echoes absolute server paths; unsanitised `filename` (`../` escapes temp
  dir); same in `git_push` (:598-600).
- W-C5 live_dashboard/views.py:141-167: `cursor.execute(sql)` with only a
  startswith("select") check + substring blocklist — dumps any table.
- W-C6 integrations sandbox: `allowed_commands` defaults to empty (falsy →
  no restriction), then `container.exec_run(command)`; `provision_sandbox`
  runs user-supplied `docker_image`; serializer `fields="__all__"` lets the
  caller control docker_image, network_access, allowed_commands, status,
  container_id.
- W-C7 experiment/views.py:499-507: `os.path.join(base, artifact_path)`,
  no containment or ownership check — `../` escapes (SQLite DB, .env).
- W-C8 gpu_escrow/views.py:44,48-60: DEPOSIT is a mock that credits real
  balance (`balance_usd += amount_usd`, models.py:40-42) with no payment.
- W-C22 paper/consumers.py:167-190: CollaborationConsumer unconditionally
  group_add + accept(); broadcasts file_edit/cursor/chat with spoofed
  usernames. (Sibling PaperCollabConsumer correctly closes 4001/4003.)
- W-H46 gpu_marketplace/consumers.py:26-28,40-68: connect() accepts WITHOUT
  any authentication; handles only register/system_stats/gpu_snapshot/
  job_update — agent's heartbeat/container_stats/container_logs and expected
  heartbeat_ack/pong/benchmark_request/config_update are silently dropped.
- W-H28 notes webhook test: `requests.post(url, ...)` with no allowlist/
  DNS/private-IP check; any authenticated user can POST to any URL incl.
  internal/metadata addresses, with HMAC attached.
- W-H12 mlops/views.py:40: `requests.post(endpoint.endpoint_url, ...)`
  with a user-created URL, no allowlist, no timeout; returns resp.json()
  directly (500 on non-JSON).
- W-M22 paper/views.py:587-660: destructive `git push -u origin main
  --force` with user-controlled remote URL; `git_pull` clones arbitrary
  remotes and `open()`s every file <5 MB into PaperFile.
- W-H13 gpu_validator/views.py:55-58: stake is self-declared, only a >=$10
  check; comment admits "In production: verify the user actually has the
  funds".

## Tasks
1. Terminal WS: close anonymous connections (4001) BEFORE pty.fork();
   require staff; move token to header/AuthMiddlewareStack; per-session
   channel group; nonzero idle timeout. Add a test asserting an anonymous
   connect is closed (terminal/tests.py never tests it — W-L6).
2. LaTeX: require auth + rate-limit + timeout + size limit; always
   `-no-shell-escape`; filename allowlist `^[A-Za-z0-9_.\-]+$`; delete the
   fake success log. `load_paper_source` must not echo absolute paths.
3. Raw SQL endpoint: REMOVE it (or replace with a fixed allowlist of
   parameterised queries as a read-only DB user, hard row limit, staff-only
   — record which you chose and why).
4. Docker sandbox: staff-only; hard-coded allowlisted images server-side;
   mandatory non-empty `allowed_commands`; remove docker_image,
   network_access, allowed_commands, status, container_id from writable
   serializer fields; no network_access=True without approval.
5. Artifact download: `Path(base, artifact_path).resolve()` + reject outside
   `base`; require run ownership/sharing.
6. Escrow deposit: DISABLE (or staff-only test flag) until a real payment
   webhook verifies each deposit; never trust client-supplied amounts.
7. Paper collab WS: close unauthenticated (4001), check project access via
   `_check_access()` before accept(); remove the duplicate `ws/collab/<uuid>/`
   registration.
8. Agent provider WS: authenticate connect() (validate token + listing
   ownership); add server handlers for heartbeat/container_stats/
   container_logs and send heartbeat_ack/pong/benchmark_request/
   config_update as the agent expects — or remove the unhandled types from
   the agent (record which).
9. Webhooks + mlops SSRF: validate URLs (public http(s) only, block
   RFC-1918/loopback/link-local AFTER DNS resolution, allowlist if
   possible); short timeout; rate-limit the test endpoint; graceful errors.
10. Git: drop `--force` (or explicit confirmation); allowlist/scheme-check
    remotes; skip binary/unreadable files; validate filenames.
11. Validator stake: check against the real gpu_escrow balance and lock
    funds via `EscrowAccount.lock_funds()` before creating an active node.

## Constraints / do-not-do
- Do NOT break the legitimate flows (staff terminal, staff LaTeX compile,
  staff sandbox) — gate them, don't delete them unless the task says REMOVE.
  Never weaken an existing check. Do not change any file outside the scope.
  Do not commit anything.

## Tests to add/run
- New tests: anonymous terminal WS connect → closed; unauthenticated
  compile-pdf POST → 401; raw-SQL POST → 404/405; escrow deposit with an
  arbitrary amount → rejected; agent WS connect without token → rejected;
  artifact download with `../` → rejected. Run the affected apps' existing
  tests and report counts.

## Verification (agent MUST run and report)
- `curl`/WS spot-checks from the audit §E cmd 25 (after fixes): terminal WS
  without token → closed; compile-pdf unauthenticated → 401; raw SQL → 404;
  escrow deposit → rejected. Paste the outputs.
- `python manage.py check` output, pasted.
- `git diff --stat` and `git status --short` — confirm only the listed files changed.
- Checklist: [ ] terminal gated [ ] LaTeX gated + fake log gone [ ] raw SQL removed/replaced [ ] sandbox allowlisted [ ] artifacts contained [ ] escrow disabled [ ] paper WS authed [ ] agent WS authed + handlers [ ] webhooks/mlops validated [ ] git force dropped [ ] stake verified.

## Definition of done
- Every surface above is closed, staff-gated, or removed, with a test
  proving the anonymous/unauthenticated path fails.

## Report back
- Per surface: closed / gated / removed + exact file:line.
- Any legitimate flow you had to change and how it still works for staff.
- Any new finding.

## Stop conditions
- If a fix would break a legitimate staff flow with no gated alternative —
  STOP for that item and report it; do not silently delete the feature.
- If a test that passed before now fails — STOP, do not improvise, report
  exactly what is blocked.
```

---

