# AGT-04 Verification & Implementation Report: Serving Security Defaults & HF Token Secret Protection

**Task ID:** AGT-04 (P0, Findings F-88 and F-89)  
**Package:** `neuralops` (`v2.0.6`)  
**Environment:** Existing project venv `/Users/tahamajs/dev/Project/.venv/bin/python` (Python 3.14.3)  
**Date:** 2026-11-07  

---

## 1. Executive Summary

This task resolves two interrelated P0 security vulnerabilities:
1. **Serving & Web UI Defaults (F-89):** Serving previously defaulted to binding `0.0.0.0` with zero authentication and unrestricted CORS (`allow_origins=["*"]`), while `--share` did not enforce `--auth`. The default host has been updated to `127.0.0.1` across `api.serve` and `neuralops serve`, non-localhost binding without an API key fails closed, HTTP Bearer authentication and per-IP sliding-window rate limiting (default 60 req/min) were added to the OpenAI-compatible serving adapter (`/health` exempt), UI CORS was locked down to localhost origins (`http://127.0.0.1:8080`, `http://localhost:8080`), `--share` now strictly requires `--auth`, password ingestion via argv (`--auth USER:PASS`) was rejected in favor of `NEURALOPS_UI_AUTH` or `getpass`, and the dead no-op `--read-write` flag was removed.
2. **HF Token Leak via Serialisable Config (F-88):** Dataclass serialization previously exposed `hf_token` as plain text through `BenchmarkConfig.to_dict()` and `save()`, as well as on `effective_config` and `--hf-token` CLI argv. `hf_token` has been removed from `BenchmarkConfig.to_dict()` and `save()`, redacted from `BenchmarkConfig.__repr__()`, and the `--hf-token` CLI argument has been deprecated and refused with a clear security error requiring `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` environment variables instead.

---

## 2. Files Changed and Created

### Files Modified:
- `neuralops/api.py`: Updated `api.serve` default `host="127.0.0.1"`, added `api_key` and `rate_limit_rpm` parameters with fail-closed non-localhost validation, and documented rate limiting.
- `neuralops/cli/serve_cmd.py`: Changed `--host` default to `"127.0.0.1"`, added `--api-key` argument (reading fallback from `NEURALOPS_API_KEY`), and enforced API key requirement for non-localhost addresses.
- `neuralops/integrations/serving.py`: Added HTTP Bearer API key authentication and sliding-window per-IP rate limiting middleware to `create_openai_serving_app`, with `/health` exempt.
- `neuralops/ui/server.py`: Restricted CORS `allow_origins` from wildcard `["*"]` to `["http://127.0.0.1:8080", "http://localhost:8080"]`.
- `neuralops/ui/cli.py`: Enforced `--auth` requirement when `--share` is enabled (raises `ValueError`), removed insecure argv password format (`--auth USER:PASS`) in favor of `--auth USERNAME` reading from `NEURALOPS_UI_AUTH` or `getpass`, removed the dead no-op `--read-write` flag, and deferred tunnel setup until after credentials validate.
- `neuralops/benchmarks/config.py`: Overrode `to_dict()` to strip `hf_token`, overrode `__repr__()` to redact `hf_token`, and ensured `save()` writes JSON without credentials; added security trace code comment.
- `neuralops/benchmarks/cli.py`: Deprecated and refused `--hf-token` argument with `_DeprecatedHfTokenAction`, resolved `hf_token` from environment variables (`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`), and prevented storing credentials in `args.effective_config`.
- `README.md`: Updated serving examples from `0.0.0.0` to `127.0.0.1`.
- `CHANGELOG.md`: Documented all breaking changes under version `[2.0.6]`.

### Files Created:
- `tests/test_serve_security.py`: Unit tests for default host, fail-closed non-localhost binding, protected endpoint authentication, and unauthenticated `/health` probe.
- `tests/ui/test_ui_cors.py`: Unit tests verifying CORS is restricted to localhost (no wildcard) and `--share` requires `--auth`.
- `tests/benchmarks/test_hf_token_secret.py`: Unit tests confirming `to_dict()`, `save()`, and `repr()` exclude plaintext token, and `--hf-token` CLI flag is refused.

---

## 3. Task 1 Trace Result (HF Token Leak Investigation)

### Detailed Trace Path:
1. `BenchmarkConfig.to_dict()` originally executed `asdict(self)`. Because `hf_token: Optional[str] = None` was a standard field on the dataclass, any configured token was included in the dictionary representation.
2. `BenchmarkConfig.save()` persisted `json.dumps(self.to_dict(), indent=2)` directly to disk.
3. `neuralops/results_saver/pipeline/result_writer.py`:
   - Line 42 (`save_config`): Called `cfg.to_dict()` and wrote `config.json` to the experiment run directory on disk.
   - Line 176 (`write_run_summary`): Called `cfg.to_dict()` and formatted its key-value pairs into `summary.md` on disk.
4. `neuralops/benchmarks/kd/runner.py`:
   - Line 343: Serialized `"config_snapshot": self.config.to_dict()` into the distillation run manifest/journal on disk.
5. `neuralops/benchmarks/cli.py`:
   - Line 674 (`apply_config_to_args`): Assigned `args.effective_config = config.to_dict()`.
   - Line 199 & 826: `--hf-token` argument took a raw token on argv (visible in `ps` and shell history) and passed it to `Runner` / `BenchmarkConfig`.

### Physical Repo Audit Findings:
- Exhaustive search across `results/` for `hf_` token patterns confirmed **no active token-carrying dict or physical user token is currently persisted on disk** in existing committed run artifacts (matches were limited to empty fields like `"hf_repo_id": ""` and test string regexes).
- **Physical Leak Status:** NOT confirmed in existing checked-in repository result files.
- **Vulnerability Status:** CONFIRMED HIGH RISK. The code paths unconditionally wrote the token to disk whenever `hf_token` was supplied by an operator or CLI user.
- **Remediation:** Dropping `hf_token` from `BenchmarkConfig.to_dict()` and `save()`, redacting `repr(config)`, and refusing argv `--hf-token` closes all downstream write sites (`config.json`, `summary.md`, run manifests, and effective config).

---

## 4. Test Verification & Pre-Fix Failure Rationale

All 9 new tests in the 3 test suites were run prior to the fix and confirmed to fail; all 9 passed cleanly after the implementation.

| Test File | Test Name | Result | Pre-Fix Failure Reason |
|:---|:---|:---:|:---|
| `tests/test_serve_security.py` | `test_default_host_is_localhost` | **PASSED** | Default host was `0.0.0.0` in both `api.serve` and `serve_cmd`. |
| `tests/test_serve_security.py` | `test_non_localhost_without_api_key_fails` | **PASSED** | `api.serve(host="0.0.0.0")` did not raise an exception without an API key. |
| `tests/test_serve_security.py` | `test_api_key_required_for_protected_endpoints` | **PASSED** | `create_openai_serving_app` had no auth middleware; endpoints returned 200 without auth. |
| `tests/test_serve_security.py` | `test_health_exempt_from_auth` | **PASSED** | `create_openai_serving_app` did not accept `api_key` keyword argument (`TypeError`). |
| `tests/ui/test_ui_cors.py` | `test_cors_not_wildcard` | **PASSED** | `CORSMiddleware` configured `allow_origins=["*"]`. |
| `tests/ui/test_ui_cors.py` | `test_share_requires_auth` | **PASSED** | `cmd_ui` with `share=True` and `auth=None` did not raise `ValueError` (attempted tunnel). |
| `tests/benchmarks/test_hf_token_secret.py` | `test_to_dict_excludes_token` | **PASSED** | `cfg.to_dict()` and `repr(cfg)` contained plaintext `hf_xxx_secret`. |
| `tests/benchmarks/test_hf_token_secret.py` | `test_save_excludes_token` | **PASSED** | `cfg.save()` wrote plaintext `hf_xxx_secret` into JSON file on disk. |
| `tests/benchmarks/test_hf_token_secret.py` | `test_no_hf_token_cli_flag` | **PASSED** | `_build_parser()` accepted `--hf-token` without warning or refusal. |

**Pass count summary:**
- `tests/test_serve_security.py`: 4 passed (100%)
- `tests/ui/test_ui_cors.py`: 2 passed (100%)
- `tests/benchmarks/test_hf_token_secret.py`: 3 passed (100%)
- **Total:** 9 passed in 6.99s.

### Regression Test Suite:
- `tests/test_packaging_wheel_cli.py`: 2 passed
- `tests/test_train_cmd_fallback.py`: 3 passed
- `tests/benchmarks/test_trust_remote_code.py`: 3 passed
- `tests/ui/test_auth_tunnel.py`: 6 passed
- **Regression Total:** 14 passed in 10.94s.

---

## 5. Manual `curl` Check Verification

A live server was started locally on `127.0.0.1:8000` via `api.serve(torch.nn.Linear(2, 2), port=8000, host="127.0.0.1", api_key="testkey")`:
- **Unauthenticated probe:**
  ```bash
  curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/v1/models
  ```
  **Observed Output:** `401`
- **Authenticated probe:**
  ```bash
  curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer testkey" http://127.0.0.1:8000/v1/models
  ```
  **Observed Output:** `200`
- **Health check probe:**
  ```bash
  curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health
  ```
  **Observed Output:** `200`

---

## 6. Static Analysis & Verification Invariants

1. **Check 0.0.0.0 default host occurrences:**
   ```bash
   grep -rn '0\.0\.0\.0' neuralops/api.py neuralops/cli/serve_cmd.py
   ```
   **Output:** Zero occurrences (exit code 1).
2. **Check hf-token in CLI:**
   ```bash
   grep -rn "hf-token" neuralops/benchmarks/cli.py
   ```
   **Output:** Only the deprecation/refusal message at lines 203 & 209.
3. **Check token serialization:**
   ```bash
   python -c "from neuralops.benchmarks.config import BenchmarkConfig; c=BenchmarkConfig(hf_token='hf_SECRET123'); print('SECRET123' in str(c.to_dict()))"
   ```
   **Output:** `False`
4. **Check UI CORS allow_origins:**
   ```bash
   grep -rn "allow_origins" neuralops/ui/server.py
   ```
   **Output:** `allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],`
5. **Ruff Linter Check:**
   ```bash
   python -m ruff check neuralops/api.py neuralops/cli/serve_cmd.py neuralops/integrations/serving.py neuralops/ui/server.py neuralops/ui/cli.py neuralops/benchmarks/config.py neuralops/benchmarks/cli.py
   ```
   **Output:** `All checks passed!`

---

## 7. Git Status and Diff Stat

### `git status --short`
```text
 M CHANGELOG.md
 M README.md
 M neuralops/api.py
 M neuralops/benchmarks/cli.py
 M neuralops/benchmarks/config.py
 M neuralops/cli/serve_cmd.py
 M neuralops/integrations/serving.py
 M neuralops/ui/cli.py
 M neuralops/ui/server.py
?? .agents/agt-prompts/
?? .agents/agt-reports/
?? tests/benchmarks/test_hf_token_secret.py
?? tests/test_serve_security.py
?? tests/ui/test_ui_cors.py
```

### `git diff --stat`
```text
 CHANGELOG.md                      |  3 ++
 README.md                         |  4 +--
 neuralops/api.py                  | 17 ++++++++++--
 neuralops/benchmarks/cli.py       | 23 ++++++++++++++--
 neuralops/benchmarks/config.py    | 24 +++++++++++++++-
 neuralops/cli/serve_cmd.py        | 22 ++++++++++++---
 neuralops/integrations/serving.py | 58 ++++++++++++++++++++++++++++++++++++++-
 neuralops/ui/cli.py               | 46 ++++++++++++++++---------------
 neuralops/ui/server.py            |  2 +-
 9 files changed, 164 insertions(+), 35 deletions(-)
```

---

## 8. Checklist of Requirements

- [x] Default host is `127.0.0.1` in both `api.serve` and `neuralops/cli/serve_cmd.py`
- [x] Non-localhost without API key fails closed with `ValueError`
- [x] `/health` exempt from authentication and rate limiting
- [x] CORS not wildcard (`["http://127.0.0.1:8080", "http://localhost:8080"]`)
- [x] `--share` requires `--auth` (raises `ValueError` if absent)
- [x] Password not readable from argv (`--auth USER:PASS` refused, reads from `NEURALOPS_UI_AUTH` or `getpass`)
- [x] `--read-write` dead flag removed
- [x] Token absent from `to_dict()`, `save()`, logs, `__repr__()`, and CLI argv
- [x] All 9 new tests pass
- [x] All 14 regression tests pass
- [x] Breaking changes documented in `CHANGELOG.md` under `[2.0.6]`

---

## 9. Limitations & Deferred Work

- **Live Cloudflare Tunnel Execution:** Live external tunnel establishment was not tested over real networks to prevent external socket exposure in the test environment; mock lifecycle verification remains authoritative via `tests/ui/test_auth_tunnel.py`.
- **GPU Invariants:** GPU/HPC kernel execution was not run on local macOS per `AGENTS.md` §4.1.
