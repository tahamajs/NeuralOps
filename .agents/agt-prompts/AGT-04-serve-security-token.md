You are working in /Users/tahamajs/dev/Project (the NeuralOps repo root; the
package lives in /Users/tahamajs/dev/Project/neuralops). Only change the files
listed. Do not touch anything else.

## Role + scope
You are fixing two linked security problems in one prompt: (a) serving and the
web UI are open by default, and (b) the HF token flows through a serialisable
config. Your scope is the serve path, the UI server/CLI/auth files, the benchmark
config and CLI token handling, the files that consume the serialised config, and
the three new test files listed below. Do not redesign serving, do not change
model-loading logic, and do not touch the `trust_remote_code` default (that is
AGT-03).

## Background
Exact problems, with file paths and line references (audit, 2026-11-06):

SERVING (F-89):
- `neuralops/api.py:275-303`: `api.serve(host="0.0.0.0", port=8000)`.
- `neuralops/cli/serve_cmd.py`: `--host` defaults to `0.0.0.0`. Serve examples in
  `README.md` also use `0.0.0.0`.
- A grep for `Authorization|api_key|APIKey|CORSMiddleware|allow_origins|Depends|
  HTTPBearer` across `neuralops/integrations/serving.py`,
  `neuralops/export/serving.py` and `neuralops/cli/serve_cmd.py` returned ZERO
  hits: there is no auth and no rate limiting in those files. The repo's own rule
  (`AGENTS.md` §15) says monitoring binds to localhost by default — serving does
  the opposite.
- `neuralops/ui/server.py:52-58`: the comment says "Restrict CORS to localhost"
  but the code sets `allow_origins=["*"]`, `allow_credentials=True`,
  `allow_methods=["GET", "POST"]`, `allow_headers=["*"]`.
- `neuralops/ui/auth.py`: Basic Auth exists and correctly uses
  `secrets.compare_digest`, and exempts `/health` — keep it, do not rewrite it.
- `neuralops/ui/cli.py`: `--auth USER:PASS` is visible in `ps` / shell history;
  `--share` publishes the UI through a Cloudflare tunnel
  (`neuralops/ui/tunnel.py`, `cloudflared`, 15-second timeout, real
  implementation) and `--auth` is documented as required with `--share` but the
  code (`cmd_ui`) only validates the format and does NOT enforce it.
- `neuralops/ui/cli.py`: `--read-write` is a no-op — it only prints
  "[INFO] --read-write flag noted: Full execution controls deferred to later
  phases."

HF TOKEN (F-88):
- `neuralops/benchmarks/config.py:70`: `hf_token` is a plain field on
  `BenchmarkConfig`; `BenchmarkConfig.to_dict()` is `dataclasses.asdict` (so it
  includes the token); `BenchmarkConfig.save()` writes `to_dict()` as JSON.
- `neuralops/benchmarks/cli.py:663`: `args.effective_config = config.to_dict()`
  is stored; `neuralops/benchmarks/cli.py:188`: `--hf-token` exposes the token on
  the command line (visible in shell history / `ps`); `cli.py:~805` passes it to
  the `Runner` / `BenchmarkConfig`.
- It was NOT traced whether a token-carrying dict is actually written to disk or
  printed to stdout — treat this as high-risk, NOT as a confirmed leak. Do NOT
  claim in any file or report that a leak did or did not happen until the trace
  in Task 1 is complete.

## Tasks
1. TRACE FIRST (before changing token handling): follow `to_dict()`, `save()`
   and `effective_config` through `neuralops/benchmarks/cli.py` (lines 663,
   ~805), `neuralops/benchmarks/orchestrator.py` and `neuralops/results_saver/`
   (search for `effective_config`, `to_dict()` and `save()`). Write down, in your
   report and as a code comment where the fix lands, whether a token-carrying
   dict is actually written to disk or printed to stdout/log. If a leak IS
   confirmed, say so explicitly in your report (and note that any exposed tokens
   must be rotated by their owner — you cannot rotate them yourself). If it is
   NOT confirmed, say that explicitly too. Either way, do Tasks 2–4: the risk is
   high regardless.
2. Remove the token from every serialisable / observable surface:
   - In `neuralops/benchmarks/config.py`, override `to_dict()` so it NEVER
     includes `hf_token` (drop the key, or replace it with `"***REDACTED***"` —
     pick one and use it consistently), and make `save()` never write the token
     to JSON. Prefer storing the token as a `SecretStr` (Pydantic) if the file
     already uses Pydantic; otherwise keep a plain field but ensure
     `dataclasses.asdict(config)` is never the thing that gets serialised —
     `to_dict()` is the only serialisation path and it excludes the token — and
     ensure `__repr__` does not show the token.
   - No log line may print the token, even partially. Grep for every place the
     token is logged and remove it.
   - In `neuralops/benchmarks/cli.py`, remove the `--hf-token` flag (line 188).
     The token is read ONLY from the environment (`HF_TOKEN`, falling back to
     `HUGGING_FACE_HUB_TOKEN`), never from argv. If removing the flag outright
     breaks a documented workflow, keep it ONLY as a deprecated flag that prints
     a warning telling the user to use `HF_TOKEN` and refuses the value — it must
     never accept or store a token. Record whichever you chose in `CHANGELOG.md`.
   - Fix every consumer found in Task 1 (`orchestrator.py`, `results_saver/`,
     and the `effective_config` assignment at `cli.py:663`) so no token-carrying
     dict is stored, written, or logged there either.
3. Secure the serve defaults:
   - Change the default host to `127.0.0.1` in BOTH `neuralops/api.py:275-303`
     (`api.serve(host="127.0.0.1", ...)`) and `neuralops/cli/serve_cmd.py`
     (`--host` default `127.0.0.1`). Change the serve examples in `README.md`
     from `0.0.0.0` to `127.0.0.1` as well.
   - Add API-key auth in `neuralops/integrations/serving.py`
     (`create_openai_serving_app`): a dependency (HTTPBearer / API key check)
     that, when an API key is configured, checks every request except `/health`.
     `api.serve()` and `cli/serve_cmd.py` must accept an `api_key` / `--api-key`
     parameter. When the host is non-localhost and no API key is configured,
     FAIL with a clear error (preferred) — do not just warn. Read the API key
     from the environment variable `NEURALOPS_API_KEY`, never from argv.
     `/health` stays exempt from auth (matching `neuralops/ui/auth.py`).
   - Restrict UI CORS in `neuralops/ui/server.py:52-58`: set `allow_origins` to
     `["http://127.0.0.1:8080", "http://localhost:8080"]` (or remove CORS
     entirely — the app is same-origin and does not need it; pick one and record
     it). `allow_credentials` may only be `True` with allow-listed origins.
     Make the comment match reality.
   - Enforce `--auth` with `--share` in `neuralops/ui/cli.py` (`cmd_ui`): when
     `--share` is given without `--auth`, fail with a clear error. Read the
     password from stdin (via `getpass`) or from the environment variable
     `NEURALOPS_UI_AUTH`, never from argv — change `--auth USER:PASS` accordingly
     and document the new form in `CHANGELOG.md` as a breaking change.
   - Either implement `--read-write` for real or remove the flag from
     `neuralops/ui/cli.py` and delete the "[INFO] --read-write flag noted..."
     print. A flag that only prints and does nothing must not survive.
4. Add basic rate limiting in `neuralops/integrations/serving.py` (a simple
   per-IP requests-per-minute limit, documented in `docs/` if a serving doc
   exists — check first; if no doc exists, document it in the `serve` docstring
   instead). If a full implementation does not fit this prompt, implement the
   simple limit anyway; do not leave it as an undocumented TODO.
5. Create the three test files (names exact):
   - `tests/test_serve_security.py`:
     `test_default_host_is_localhost` (assert `api.serve` and `serve_cmd` default
     to `127.0.0.1`); `test_non_localhost_without_api_key_fails` (assert
     `serve(host="0.0.0.0")` without an API key raises);
     `test_api_key_required_for_protected_endpoints` (with `TestClient`, assert a
     request without an `Authorization` header to a protected endpoint returns
     401 and with the correct key returns 200);
     `test_health_exempt_from_auth` (assert `/health` works without auth).
   - `tests/ui/test_ui_cors.py` (new file under `tests/ui/`):
     `test_cors_not_wildcard` (assert `allow_origins` does not contain `"*"` and
     only lists localhost origins); `test_share_requires_auth` (assert `cmd_ui`
     with `--share` and without `--auth` raises).
   - `tests/benchmarks/test_hf_token_secret.py`:
     `test_to_dict_excludes_token` (with `hf_token="hf_xxx_secret"`, assert
     `"hf_xxx_secret" not in json.dumps(config.to_dict())` and not in
     `repr(config)`); `test_save_excludes_token` (save to a temp file, assert the
     file does not contain the token); `test_no_hf_token_cli_flag` (assert
     `--hf-token` is not in the parser, or only warns and refuses the value).
   Each test must FAIL before the fix and PASS after it — state this explicitly
   in your report for each test.

## Constraints / do-not-do
- No silent fallbacks, no new dependencies, no breaking change without a
  migration note in `CHANGELOG.md`. (This prompt HAS breaking changes: the serve
  default host, the `--auth` form, `--hf-token` removal, `--read-write` removal
  if removed. List every one in `CHANGELOG.md` under the current version, 2.0.6.)
- Do not edit generated files, do not commit secrets. Never put a real token or
  password in a test, a log message, or `CHANGELOG.md` — tests use fake values
  such as `"hf_xxx_secret"`.
- Do not change which models serve, do not change serving behaviour for
  localhost requests without an API key (they must keep working), and do not
  rewrite `neuralops/ui/auth.py` (it already uses `secrets.compare_digest`
  correctly — review it, keep it).
- Do not claim a token leak happened (or did not happen) beyond what the Task 1
  trace actually established.

## Tests to add/run
- Create: `tests/test_serve_security.py`, `tests/ui/test_ui_cors.py`,
  `tests/benchmarks/test_hf_token_secret.py` (tests named in Task 5).
- Run: `python -m pytest tests/test_serve_security.py tests/ui/test_ui_cors.py tests/benchmarks/test_hf_token_secret.py -v`
- Run: `grep -rn '0\.0\.0\.0' neuralops/api.py neuralops/cli/serve_cmd.py` —
  expected: no default-host occurrence remains.
- Run: `grep -rn "hf-token" neuralops/benchmarks/cli.py` — expected: zero results,
  or only a deprecation message.
- Run: `python -c "from neuralops.benchmarks.config import BenchmarkConfig; c=BenchmarkConfig(hf_token='hf_SECRET123'); print('SECRET123' in str(c.to_dict()))"` — expected: `False`.
- Run: `python -m ruff check neuralops/api.py neuralops/cli/serve_cmd.py neuralops/integrations/serving.py neuralops/ui/server.py neuralops/ui/cli.py neuralops/benchmarks/config.py neuralops/benchmarks/cli.py`
- Run (manual, only if a server can be started locally): start serving on
  `127.0.0.1` and run
  `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/v1/models` —
  expected `401` without a key. If you cannot run this, say so explicitly; the
  `TestClient` tests are not a substitute for this check, and you must not claim
  you ran it if you did not.

## Verification (agent MUST run and report)
- `grep -rn '0\.0\.0\.0' neuralops/api.py neuralops/cli/serve_cmd.py` — expected:
  no default host remains (examples in README already changed to `127.0.0.1`).
- `python -m pytest tests/test_serve_security.py tests/ui/test_ui_cors.py tests/benchmarks/test_hf_token_secret.py -v` — expected: all tests pass; report
  pass counts per file.
- The token check from Tests above prints `False`.
- `grep -rn "allow_origins" neuralops/ui/server.py` — expected: no `"*"`.
- `git diff --stat` and `git status --short` — confirm only the files listed in
  Tasks 2–5 changed (plus `README.md`, `CHANGELOG.md`, and the rate-limit doc /
  docstring).
- Checklist to tick: [ ] default host is `127.0.0.1` in both places
  [ ] non-localhost without API key fails [ ] `/health` exempt [ ] CORS not
  wildcard [ ] `--share` requires `--auth` [ ] password not readable from argv
  [ ] `--read-write` implemented or removed [ ] token absent from `to_dict()` /
  `save()` / logs / argv [ ] all new tests pass.

## Definition of done
- `api.serve` and `serve_cmd` default to `127.0.0.1`; serving on `0.0.0.0`
  without an API key raises.
- Protected endpoints return 401 without a key and 200 with the correct key;
  `/health` works without auth.
- UI CORS `allow_origins` contains no `"*"` and lists only localhost origins.
- `cmd_ui` with `--share` and without `--auth` raises; the password is read from
  stdin or `NEURALOPS_UI_AUTH`, never from argv.
- `--read-write` is implemented or gone (no no-op print remains).
- `BenchmarkConfig(hf_token="hf_xxx_secret").to_dict()` and `save()` output
  contain no trace of the token; `--hf-token` no longer accepts a token.
- All new tests pass; each would have failed before the fix.
- `CHANGELOG.md` lists every breaking change from this prompt.

## Report back
- Files changed (exact paths).
- The Task 1 trace result: is a token-carrying dict actually written to disk or
  printed anywhere? Quote the evidence. If a leak is confirmed, say so plainly
  (tokens must be rotated by their owner).
- Test results with pass/fail counts per file, and for each test why it would
  have failed before the fix.
- Whether you ran the manual `curl` check, and its result — or the explicit
  statement that you did not run it.
- What you could NOT verify.
- Any new finding.

## Stop conditions
- If `neuralops/api.py:275-303`, `neuralops/cli/serve_cmd.py`,
  `neuralops/ui/server.py:52-58`, `neuralops/benchmarks/config.py:70` or
  `neuralops/benchmarks/cli.py:188` do not match the description — STOP, quote
  what you actually found, and report it.
- If a test that passed before (per `BASELINE.md`) now fails — STOP and report
  which test and its failure output.
- If the token trace (Task 1) shows the token IS being written somewhere you
  cannot fix within the listed files — STOP and report the exact write site;
  do not improvise a fix outside the listed files.
- If a decision is needed (e.g. remove CORS entirely vs. allow-list it, remove
  `--read-write` vs. implement it) — STOP and report the decision needed instead
  of choosing silently; the defaults if you must proceed are: allow-list CORS
  (do not remove it) and remove `--read-write`.
