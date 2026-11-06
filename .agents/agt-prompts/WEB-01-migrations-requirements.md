## WEB-01 — Migrations for all 61 Django apps + requirements that install what the code imports — Phase P0 — Findings W-C1, W-C24, W-M1

**Depends on:** — (no dependency; run first — unblocks everything)
**Goal:** Every one of the 61 apps has migrations in sync with its models (`makemigrations --check --dry-run` prints "No changes detected"), `manage.py check` passes, and a fresh install from `requirements.txt` can import settings.
**Context (Persian, 2-4 lines):** ‏۴۸ اپ از ۶۱ اپ migration ندارند، پس جدول‌هایشان وجود ندارد؛ و `requirements.txt` چیزهایی را که کد import می‌کند نصب نمی‌کند، پس نصب تازه حتی settings را import نمی‌کند. این پرامپت اول زیرساخت اجرا را درست می‌کند.
**Prompt to paste (English):** ```
You are working in /Users/tahamajs/Documents/uni/Project_website (the website
repo root; backend in django_api/). Only change the files listed. Do not touch
anything else. Use ONLY the existing project venv — never create a new venv.
Do not commit anything.

## Role + scope
You are a Django backend engineer. Your scope is `django_api/**/models.py`
(all 61 apps), `django_api/**/migrations/` (create where missing),
`django_api/requirements.txt` (+ a lockfile if you create one),
`django_api/django_api/settings.py` (DATABASE config only), and
`django_api/.gitignore` (db.sqlite3 entry only). Run your own
multi-sub-agents in parallel for the independent sub-tasks: (a) one sub-agent
runs makemigrations per app batch (A–M, N–Z) and reviews each migration;
(b) one sub-agent regenerates requirements from the real environment
(pip freeze) and diffs it against the imports; (c) one sub-agent verifies
the DB config + gitignore. You synthesize their reports into ONE final report.

## Background
Exact problems (audit W-C1, W-C24, W-M1, 2026-11-07):
- 48 of 61 apps define models but ship zero migrations; only agents, api,
  experiment, failure_forensics, gpu_marketplace, jobscheduler,
  model_versioning, notes, profiles, research, support, terminal,
  web_dashboard have migrations. E.g. `ai_coscientist/models.py:7` defines
  `ResearchSession`, `ai_coscientist/views.py:27` queries it, no migrations/.
- `requirements.txt` (36 lines, all `>=` ranges, no lockfile) lacks celery,
  django-allauth, redis (settings.py imports `celery.schedules.crontab` at
  :12, lists allauth at :94-97, uses `channels_redis` at :198-225), and N-Z
  apps import openai, huggingface_hub, weasyprint, langchain_*/langgraph,
  optuna, duckduckgo_search, chromadb — none listed. `support/tasks.py`
  uses removed `openai.ChatCompletion.create` (must migrate to the
  `openai>=1.0` client). Conversely it lists unused python-decouple,
  djangorestframework-simplejwt, django-extensions.
- SQLite `db.sqlite3` (7.5 MB) is the only DB; no prod DB path; live DB in
  repo root.

## Tasks
1. For EACH of the 48 apps without migrations: run `makemigrations <app>`,
   read the generated migration, and confirm it matches the models (no
   spurious drops/renames). Then run `migrate` against a DISPOSABLE COPY of
   the DB — never against the live `db.sqlite3` — and record the result.
   Add `makemigrations --check --dry-run` to CI (or to a script the CI can
   call) so drift fails the build.
2. Regenerate `requirements.txt` from the real environment (`pip freeze`),
   pin versions (or commit a lockfile), and remove or justify the unused
   deps (python-decouple, djangorestframework-simplejwt, django-extensions
   — keep simplejwt ONLY if a later prompt wires it; record the decision).
   Migrate `support/tasks.py` off `openai.ChatCompletion.create` to the
   `openai>=1.0` client.
3. Add a `DATABASE_URL`-driven Postgres config to settings.py (SQLite
   dev-only fallback); confirm `db.sqlite3` is git-ignored and backed up.
   Do NOT change any other settings value here (security settings are WEB-03).
4. Verify with a FRESH venv-free check (existing venv only): `python
   manage.py check` passes and `python -c "import django; django.setup()"`
   succeeds with the regenerated requirements installed.

## Constraints / do-not-do
- Never run migrations or the audit scripts against the live `db.sqlite3`.
  Never invent a model field or delete a model. Do not change any view,
  serializer, URL, or security setting. Do not commit anything.

## Tests to add/run
- New CI check: `python manage.py makemigrations --check --dry-run`
  (expected: "No changes detected").
- Commands: `python manage.py check` (expected: no issues);
  `python manage.py showmigrations | grep -v '\[X\]'` (expected: no
  unapplied migrations); `python -c "import django; django.setup(); from
  django.apps import apps; print(len(list(apps.get_app_configs())))"`
  (expected: 78).

## Verification (agent MUST run and report)
- `makemigrations --check --dry-run` output, pasted.
- `showmigrations` unapplied list (expected: empty), pasted.
- `manage.py check` output, pasted.
- Per-app migration list (48 apps, file paths), pasted.
- `git diff --stat` and `git status --short` — confirm only the listed files changed.
- Checklist: [ ] 61/61 apps migrated [ ] check passes [ ] fresh import works [ ] requirements regenerated + pinned [ ] openai client migrated [ ] DATABASE_URL config added.

## Definition of done
- All 61 apps have migrations in sync with their models; `migrate` succeeds
  on a disposable DB; `manage.py check` passes; settings imports on a fresh
  install from the regenerated requirements.

## Report back
- Per-app migration files created (exact paths).
- Requirements diff summary (added / removed / pinned).
- Any model that could not be migrated and why.

## Stop conditions
- If a migration would drop a table/column with live data — STOP for that
  app and report it; do not run it.
- If `manage.py check` fails for a reason outside this scope — STOP and
  report exactly what is blocked.
- If a test that passed before now fails — STOP, do not improvise, report
  exactly what is blocked.
```

