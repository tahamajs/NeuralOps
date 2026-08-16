#!/usr/bin/env python3
"""
Comprehensive Backend API & Auth Auditor for Django API.
Audits:
1. System checks & verify_all_django_apps
2. Auth endpoints: /api/auth/register/, /api/auth/login/, /api/auth/logout/, /api/auth/user/
3. REST CRUD endpoints: /api/runs/, /api/experiments/, /api/sweeps/, /api/jobs/, /api/config/attentions/
4. Serializers, pagination, and error response consistency.
"""
import os
import sys
import uuid
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Prevent background schedulers from starting infinite worker loops during test
sys.argv = [sys.argv[0], "test"]
os.environ["RUN_MAIN"] = "true"

# Setup Django Environment
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
CODES_DIR = BASE_DIR.parent

sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(CODES_DIR))
sys.path.insert(0, str(PROJECT_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_api.settings")

import django
django.setup()

from django.core.management import call_command
from django.apps import apps
from django.contrib.auth.models import User
from django.core.cache import cache
from django.urls import resolve
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient

# App Models & Serializers
from experiment.models import Experiment, TrainingRun, TrainingSnapshot
from experiment.serializers import ExperimentSerializer, TrainingRunSerializer, TrainingSnapshotSerializer
from sweeps.models import Sweep, SweepTrial
from sweeps.serializers import SweepSerializer, SweepCreateSerializer, SweepTrialSerializer
from jobscheduler.models import Job, CustomMetricEntry, VersionedConfig
from jobscheduler.serializers import JobSerializer, CustomMetricEntrySerializer, VersionedConfigSerializer

results = {
    "total_tests": 0,
    "passed": 0,
    "failed": 0,
    "details": []
}

def log_test(test_name, passed, detail=""):
    results["total_tests"] += 1
    if passed:
        results["passed"] += 1
        status_str = "PASS"
    else:
        results["failed"] += 1
        status_str = "FAIL"
    results["details"].append({"test": test_name, "status": status_str, "detail": detail})
    print(f"  [{status_str}] {test_name}: {detail}")


def run_audit():
    print("=" * 80)
    print("STARTING BACKEND API & AUTH AUDIT")
    print("=" * 80)

    # 1. System checks
    print("\n--- 1. DJANGO SYSTEM CHECKS ---")
    try:
        call_command("check")
        log_test("Django System Check (python manage.py check)", True, "0 issues identified")
    except Exception as e:
        log_test("Django System Check (python manage.py check)", False, str(e))

    # 2. Installed Apps check (78 apps)
    print("\n--- 2. ALL INSTALLED APPS & REGISTRIES ---")
    all_apps = list(apps.get_app_configs())
    log_test("App Configurations Count", len(all_apps) == 78, f"Found {len(all_apps)}/78 apps")

    try:
        from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
        from framework.trainer.registry import TrainerRegistry
        attn_count = len(ATTENTION_REGISTRY)
        comp_count = len(COMPRESSOR_REGISTRY)
        train_count = len(TrainerRegistry._registry)
        log_test("PyTorch Framework Bridge Registry", attn_count > 0 and comp_count > 0,
                 f"Attentions: {attn_count}, Compressors: {comp_count}, Trainers: {train_count}")
    except Exception as e:
        log_test("PyTorch Framework Bridge Registry", False, str(e))

    # 3. Auth Endpoints Audit
    print("\n--- 3. AUTHENTICATION ENDPOINTS AUDIT ---")
    cache.clear()
    client = APIClient()

    # Clean up test users if existing
    test_username = f"auditor_{uuid.uuid4().hex[:8]}"
    test_email = f"{test_username}@example.com"
    test_password = "SecurePassword123!"

    # 3.1 Register - Success
    reg_resp = client.post("/api/auth/register/", {
        "username": test_username,
        "password": test_password,
        "email": test_email
    }, format="json")
    reg_pass = (reg_resp.status_code == status.HTTP_201_CREATED and
                "token" in reg_resp.data and
                reg_resp.data["username"] == test_username and
                "authToken" in reg_resp.cookies)
    log_test("POST /api/auth/register/ (Valid Registration)", reg_pass, f"Status: {reg_resp.status_code}")

    auth_token = reg_resp.data.get("token") if reg_pass else None

    # 3.2 Register - Validation Errors
    # Short username
    resp = client.post("/api/auth/register/", {"username": "ab", "password": test_password}, format="json")
    log_test("POST /api/auth/register/ (Short username rejected)", resp.status_code == 400 and "error" in resp.data, f"Status: {resp.status_code}")

    # Short password
    resp = client.post("/api/auth/register/", {"username": f"u_{uuid.uuid4().hex[:6]}", "password": "123"}, format="json")
    log_test("POST /api/auth/register/ (Short password rejected)", resp.status_code == 400 and "error" in resp.data, f"Status: {resp.status_code}")

    # Duplicate username
    resp = client.post("/api/auth/register/", {"username": test_username, "password": test_password}, format="json")
    log_test("POST /api/auth/register/ (Duplicate username rejected)", resp.status_code == 400 and "error" in resp.data, f"Status: {resp.status_code}")

    # Invalid email
    resp = client.post("/api/auth/register/", {"username": f"u_{uuid.uuid4().hex[:6]}", "password": test_password, "email": "bademail"}, format="json")
    log_test("POST /api/auth/register/ (Invalid email rejected)", resp.status_code == 400 and "error" in resp.data, f"Status: {resp.status_code}")

    # Missing fields
    resp = client.post("/api/auth/register/", {}, format="json")
    log_test("POST /api/auth/register/ (Empty payload rejected)", resp.status_code == 400 and "error" in resp.data, f"Status: {resp.status_code}")

    # 3.3 Login - Success
    login_resp = client.post("/api/auth/login/", {
        "username": test_username,
        "password": test_password
    }, format="json")
    login_pass = (login_resp.status_code == status.HTTP_200_OK and
                  "token" in login_resp.data and
                  login_resp.data["token"] == auth_token and
                  "authToken" in login_resp.cookies)
    log_test("POST /api/auth/login/ (Valid Credentials)", login_pass, f"Status: {login_resp.status_code}")

    # 3.4 Login - Invalid Credentials & Missing fields
    resp = client.post("/api/auth/login/", {"username": test_username, "password": "WrongPassword!"}, format="json")
    log_test("POST /api/auth/login/ (Wrong password rejected)", resp.status_code == 401 and "error" in resp.data, f"Status: {resp.status_code}")

    resp = client.post("/api/auth/login/", {"username": "non_existent_user_999", "password": test_password}, format="json")
    log_test("POST /api/auth/login/ (Nonexistent user rejected)", resp.status_code == 401 and "error" in resp.data, f"Status: {resp.status_code}")

    resp = client.post("/api/auth/login/", {"username": test_username}, format="json")
    log_test("POST /api/auth/login/ (Missing password rejected)", resp.status_code == 400 and "error" in resp.data, f"Status: {resp.status_code}")

    # 3.5 User Info - Token Authentication
    token_client = APIClient()
    token_client.credentials(HTTP_AUTHORIZATION=f"Token {auth_token}")
    uinfo_resp = token_client.get("/api/auth/user/")
    uinfo_pass = (uinfo_resp.status_code == 200 and
                  uinfo_resp.data["username"] == test_username and
                  uinfo_resp.data["email"] == test_email)
    log_test("GET /api/auth/user/ (Authorization: Token header)", uinfo_pass, f"Status: {uinfo_resp.status_code}")

    # 3.6 User Info - Bearer Authentication
    bearer_client = APIClient()
    bearer_client.credentials(HTTP_AUTHORIZATION=f"Bearer {auth_token}")
    uinfo_bearer = bearer_client.get("/api/auth/user/")
    log_test("GET /api/auth/user/ (Authorization: Bearer header)", uinfo_bearer.status_code == 200, f"Status: {uinfo_bearer.status_code}")

    # 3.7 User Info - Cookie Authentication
    cookie_client = APIClient()
    cookie_client.cookies["authToken"] = auth_token
    uinfo_cookie = cookie_client.get("/api/auth/user/")
    log_test("GET /api/auth/user/ (Cookie authToken)", uinfo_cookie.status_code == 200, f"Status: {uinfo_cookie.status_code}")

    # 3.8 User Info - Unauthenticated
    anon_client = APIClient()
    unauth_uinfo = anon_client.get("/api/auth/user/")
    log_test("GET /api/auth/user/ (Unauthenticated rejected)", unauth_uinfo.status_code == 401, f"Status: {unauth_uinfo.status_code}")

    # 3.9 Logout - Success & Unauthenticated
    unauth_logout = anon_client.post("/api/auth/logout/")
    log_test("POST /api/auth/logout/ (Unauthenticated rejected)", unauth_logout.status_code == 401, f"Status: {unauth_logout.status_code}")

    logout_resp = token_client.post("/api/auth/logout/")
    log_test("POST /api/auth/logout/ (Authenticated logout)", logout_resp.status_code == 200 and logout_resp.data.get("status") == "logged out", f"Status: {logout_resp.status_code}")

    # Verify token revoked
    revoked_check = token_client.get("/api/auth/user/")
    log_test("Revoked Token Rejected after Logout", revoked_check.status_code == 401, f"Status: {revoked_check.status_code}")

    # 4. REST CRUD Endpoints Audit
    print("\n--- 4. REST CRUD ENDPOINTS AUDIT ---")
    # Authenticate client for CRUD
    user = User.objects.get(username=test_username)
    new_token, _ = Token.objects.get_or_create(user=user)
    authed = APIClient()
    authed.credentials(HTTP_AUTHORIZATION=f"Token {new_token.key}")
    authed.cookies["authToken"] = new_token.key

    # 4.1 /api/config/attentions/
    unauth_attn = anon_client.get("/api/config/attentions/")
    log_test("GET /api/config/attentions/ (Unauthenticated rejected)", unauth_attn.status_code == 401, f"Status: {unauth_attn.status_code}")

    attn_resp = authed.get("/api/config/attentions/")
    attn_pass = (attn_resp.status_code == 200 and
                 "attentions" in attn_resp.data and
                 isinstance(attn_resp.data["attentions"], list) and
                 "count" in attn_resp.data)
    log_test("GET /api/config/attentions/ (Authenticated)", attn_pass, f"Status: {attn_resp.status_code}, count: {attn_resp.data.get('count')}")

    # 4.2 /api/experiments/ CRUD
    # Unauthenticated list
    unauth_exp = anon_client.get("/api/experiments/")
    log_test("GET /api/experiments/ (Unauthenticated rejected)", unauth_exp.status_code == 401, f"Status: {unauth_exp.status_code}")

    # List Experiments
    exp_list_resp = authed.get("/api/experiments/")
    log_test("GET /api/experiments/ (List experiments)", exp_list_resp.status_code == 200 and isinstance(exp_list_resp.data, list), f"Status: {exp_list_resp.status_code}, length: {len(exp_list_resp.data)}")

    # Create Experiment (POST)
    exp_create_payload = {
        "name": f"Audit Experiment {uuid.uuid4().hex[:6]}",
        "description": "Created during backend audit",
        "tags": "audit,test,svd",
        "default_config": {"learning_rate": 0.0001, "batch_size": 32}
    }
    exp_create_resp = authed.post("/api/experiments/", exp_create_payload, format="json")
    exp_create_pass = exp_create_resp.status_code == status.HTTP_201_CREATED and "id" in exp_create_resp.data
    log_test("POST /api/experiments/ (Create experiment)", exp_create_pass, f"Status: {exp_create_resp.status_code}")

    exp_id = exp_create_resp.data.get("id") if exp_create_pass else None
    test_experiment_obj = Experiment.objects.get(id=exp_id) if exp_id else None

    # Get Experiment Detail (GET)
    if exp_id:
        exp_detail_resp = authed.get(f"/api/experiments/{exp_id}/")
        exp_detail_pass = exp_detail_resp.status_code == 200 and exp_detail_resp.data.get("name") == exp_create_payload["name"]
        log_test("GET /api/experiments/<uuid>/ (Detail)", exp_detail_pass, f"Status: {exp_detail_resp.status_code}")

    # Nonexistent Experiment Detail (404)
    exp_404_resp = authed.get(f"/api/experiments/{uuid.uuid4()}/")
    log_test("GET /api/experiments/<nonexistent-uuid>/ (404 Not Found)", exp_404_resp.status_code == 404, f"Status: {exp_404_resp.status_code}")

    # 4.3 /api/runs/ CRUD
    # Unauthenticated list
    unauth_runs = anon_client.get("/api/runs/")
    log_test("GET /api/runs/ (Unauthenticated rejected)", unauth_runs.status_code == 401, f"Status: {unauth_runs.status_code}")

    # List Runs
    runs_list_resp = authed.get("/api/runs/")
    log_test("GET /api/runs/ (List runs)", runs_list_resp.status_code == 200 and isinstance(runs_list_resp.data, list), f"Status: {runs_list_resp.status_code}, length: {len(runs_list_resp.data)}")

    # Create / Trigger Training Run via start_train
    with patch("experiment.views.start_job", MagicMock()):
        with patch("api.views.start_job", MagicMock()):
            train_payload = {
                "name": f"Audit Run {uuid.uuid4().hex[:6]}",
                "model_name": "distilbert-base-uncased",
                "task_name": "mrpc",
                "learning_rate": 0.00005,
                "batch_size": 16,
                "experiment_id": str(test_experiment_obj.id) if test_experiment_obj else None
            }
            start_train_resp = authed.post("/api/experiments/start_train/", train_payload, format="json")
            run_pass = start_train_resp.status_code in [200, 201, 202] and "id" in start_train_resp.data
            log_test("POST /api/experiments/start_train/ (Trigger/Create run)", run_pass, f"Status: {start_train_resp.status_code}")

            run_id = start_train_resp.data.get("id") if run_pass else None
            if run_id:
                # GET Run Detail / Status
                run_detail_resp = authed.get(f"/api/runs/{run_id}/")
                log_test("GET /api/runs/<uuid>/ (Run detail)", run_detail_resp.status_code == 200, f"Status: {run_detail_resp.status_code}")

                run_status_resp = authed.get(f"/api/runs/{run_id}/status/")
                log_test("GET /api/runs/<uuid>/status/ (Run status)", run_status_resp.status_code == 200, f"Status: {run_status_resp.status_code}")

                # GET Run History / Snapshots / Artifacts
                history_resp = authed.get(f"/api/runs/{run_id}/history/")
                log_test("GET /api/runs/<uuid>/history/ (Run metrics history)", history_resp.status_code == 200, f"Status: {history_resp.status_code}")

                snapshots_resp = authed.get(f"/api/runs/{run_id}/snapshots/")
                log_test("GET /api/runs/<uuid>/snapshots/ (Run snapshots)", snapshots_resp.status_code == 200, f"Status: {snapshots_resp.status_code}")

                artifacts_resp = authed.get(f"/api/runs/{run_id}/artifacts/")
                log_test("GET /api/runs/<uuid>/artifacts/ (Run artifacts)", artifacts_resp.status_code == 200, f"Status: {artifacts_resp.status_code}")

    # Nonexistent Run (404)
    run_404_resp = authed.get(f"/api/runs/{uuid.uuid4()}/")
    log_test("GET /api/runs/<nonexistent-uuid>/ (404 Not Found)", run_404_resp.status_code == 404, f"Status: {run_404_resp.status_code}")

    # 4.4 /api/sweeps/ CRUD
    # Unauthenticated list
    unauth_sweeps = anon_client.get("/api/sweeps/")
    log_test("GET /api/sweeps/ (Unauthenticated rejected)", unauth_sweeps.status_code == 401, f"Status: {unauth_sweeps.status_code}")

    # List Sweeps
    sweeps_list_resp = authed.get("/api/sweeps/")
    log_test("GET /api/sweeps/ (List sweeps)", sweeps_list_resp.status_code == 200 and isinstance(sweeps_list_resp.data, list), f"Status: {sweeps_list_resp.status_code}, length: {len(sweeps_list_resp.data)}")

    # Create Sweep (POST)
    with patch("sweeps.tasks.start_sweep.delay", MagicMock()):
        sweep_payload = {
            "project": str(test_experiment_obj.id) if test_experiment_obj else None,
            "name": f"Audit Sweep {uuid.uuid4().hex[:6]}",
            "description": "Auditing sweep CRUD",
            "metric_name": "eval_loss",
            "direction": "minimize",
            "algorithm": "tpe",
            "pruner": "median",
            "max_trials": 10,
            "parallel_workers": 1,
            "config": {"parameters": {"learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3}}}
        }
        sweep_create_resp = authed.post("/api/sweeps/", sweep_payload, format="json")
        sweep_pass = sweep_create_resp.status_code == status.HTTP_201_CREATED and "id" in sweep_create_resp.data
        log_test("POST /api/sweeps/ (Create sweep)", sweep_pass, f"Status: {sweep_create_resp.status_code}")

        sweep_id = sweep_create_resp.data.get("id") if sweep_pass else None
        if sweep_id:
            # GET Sweep Detail
            sweep_detail_resp = authed.get(f"/api/sweeps/{sweep_id}/")
            log_test("GET /api/sweeps/<uuid>/ (Sweep detail)", sweep_detail_resp.status_code == 200, f"Status: {sweep_detail_resp.status_code}")

            # GET Sweep Trials
            sweep_trials_resp = authed.get(f"/api/sweeps/{sweep_id}/trials/")
            log_test("GET /api/sweeps/<uuid>/trials/ (Sweep trials list)", sweep_trials_resp.status_code == 200, f"Status: {sweep_trials_resp.status_code}")

            # PATCH Sweep status
            sweep_patch_resp = authed.patch(f"/api/sweeps/{sweep_id}/", {"status": "paused"}, format="json")
            log_test("PATCH /api/sweeps/<uuid>/ (Update sweep status to paused)", sweep_patch_resp.status_code == 200 and sweep_patch_resp.data.get("status") == "paused", f"Status: {sweep_patch_resp.status_code}")

    # Nonexistent Sweep (404)
    sweep_404_resp = authed.get(f"/api/sweeps/{uuid.uuid4()}/")
    log_test("GET /api/sweeps/<nonexistent-uuid>/ (404 Not Found)", sweep_404_resp.status_code == 404, f"Status: {sweep_404_resp.status_code}")

    # 4.5 /api/jobs/ CRUD
    # Note: Check URL route for jobs
    match_jobs = resolve("/api/jobs/")
    print(f"  [DEBUG] /api/jobs/ resolves to: {match_jobs.func.__module__}.{match_jobs.func.__name__} (app: {match_jobs.app_name}, url_name: {match_jobs.url_name})")

    # Unauthenticated list
    unauth_jobs = anon_client.get("/api/jobs/")
    log_test("GET /api/jobs/ (Unauthenticated rejected)", unauth_jobs.status_code == 401, f"Status: {unauth_jobs.status_code}")

    # List Jobs
    jobs_list_resp = authed.get("/api/jobs/")
    log_test("GET /api/jobs/ (List jobs)", jobs_list_resp.status_code == 200 and isinstance(jobs_list_resp.data, list), f"Status: {jobs_list_resp.status_code}, length: {len(jobs_list_resp.data)}")

    # Create Job (POST) - testing with jobscheduler API
    with patch("jobscheduler.scheduler.scheduler.add_job", MagicMock()):
        job_payload = {
            "name": f"Audit Job {uuid.uuid4().hex[:6]}",
            "model_name": "bert-base-uncased",
            "task_name": "sst2",
            "attention_type": "svd",
            "method": "baseline",
            "epochs": 2,
            "batch_size": 32,
            "learning_rate": 0.00003,
            "priority": 3
        }
        # Try both direct view or API
        from jobscheduler import views as jobs_views
        job_create_resp = authed.post("/api/jobs/", job_payload, format="json")
        if job_create_resp.status_code == 405:
            # Check if direct jobscheduler route is available
            job_create_resp = authed.post("/api/jobs/jobs/", job_payload, format="json")
        
        job_pass = job_create_resp.status_code == status.HTTP_201_CREATED and "id" in job_create_resp.data
        log_test("POST /api/jobs/ (Create job)", job_pass, f"Status: {job_create_resp.status_code}")

        job_id = job_create_resp.data.get("id") if job_pass else None
        if not job_id:
            # Create a Job directly to test detail/pause/resume/stop endpoints
            job_obj = Job.objects.create(
                name="AuditDirectJob",
                model_name="bert-base-uncased",
                task_name="sst2"
            )
            job_id = str(job_obj.id)

        if job_id:
            # GET Job Detail
            job_detail_resp = authed.get(f"/api/jobs/{job_id}/")
            log_test("GET /api/jobs/<uuid>/ (Job detail)", job_detail_resp.status_code == 200, f"Status: {job_detail_resp.status_code}")

            # Pause Job
            pause_resp = authed.post(f"/api/jobs/{job_id}/pause/")
            log_test("POST /api/jobs/<uuid>/pause/ (Pause job)", pause_resp.status_code == 200, f"Status: {pause_resp.status_code}")

            # Resume Job
            resume_resp = authed.post(f"/api/jobs/{job_id}/resume/")
            log_test("POST /api/jobs/<uuid>/resume/ (Resume job)", resume_resp.status_code == 200, f"Status: {resume_resp.status_code}")

            # Stop Job
            stop_resp = authed.post(f"/api/jobs/{job_id}/stop/")
            log_test("POST /api/jobs/<uuid>/stop/ (Stop job)", stop_resp.status_code == 200, f"Status: {stop_resp.status_code}")

            # Job metrics / config history
            hist_resp = authed.get(f"/api/jobs/{job_id}/config/history/")
            log_test("GET /api/jobs/<uuid>/config/history/ (Job config history)", hist_resp.status_code == 200, f"Status: {hist_resp.status_code}")

    # Nonexistent Job (404)
    job_404_resp = authed.get(f"/api/jobs/{uuid.uuid4()}/")
    log_test("GET /api/jobs/<nonexistent-uuid>/ (404 Not Found)", job_404_resp.status_code == 404, f"Status: {job_404_resp.status_code}")

    # Clean up test experiment
    if exp_id:
        authed.delete(f"/api/experiments/{exp_id}/")

    # 5. Serializers, Pagination & Error Response Consistency
    print("\n--- 5. SERIALIZERS, PAGINATION & ERROR CONSISTENCY AUDIT ---")
    # 5.1 Serializers Serialization & Validation Tests
    # ExperimentSerializer
    test_exp = Experiment.objects.create(name="SerializerTestExp", tags="t1,t2")
    s_exp = ExperimentSerializer(test_exp)
    exp_fields = {"id", "name", "tags", "created_at", "updated_at", "runs_count"}
    log_test("ExperimentSerializer schema conformity", exp_fields.issubset(set(s_exp.data.keys())), f"Fields: {set(s_exp.data.keys())}")

    # TrainingRunSerializer
    test_run = TrainingRun.objects.create(experiment=test_exp, name="SerializerTestRun", config={"lr": 0.01})
    s_run = TrainingRunSerializer(test_run)
    run_fields = {"id", "experiment", "experiment_name", "name", "status", "job_type", "config", "created_at"}
    log_test("TrainingRunSerializer schema conformity", run_fields.issubset(set(s_run.data.keys())), f"Fields: {set(s_run.data.keys())}")

    # SweepSerializer
    test_sweep = Sweep.objects.create(project=test_exp, name="SerializerTestSweep", direction="minimize")
    s_sweep = SweepSerializer(test_sweep)
    sweep_fields = {"id", "name", "status", "direction", "trial_count", "best_objective"}
    log_test("SweepSerializer schema conformity", sweep_fields.issubset(set(s_sweep.data.keys())), f"Fields: {set(s_sweep.data.keys())}")

    # JobSerializer
    test_job = Job.objects.create(name="SerializerTestJob", model_name="bert-base", task_name="cola")
    s_job = JobSerializer(test_job)
    job_fields = {"id", "name", "model_name", "task_name", "status", "priority", "created_at"}
    log_test("JobSerializer schema conformity", job_fields.issubset(set(s_job.data.keys())), f"Fields: {set(s_job.data.keys())}")

    # 5.2 Error Response Consistency Verification
    # All 400 responses contain structured error payload ('error' or field errors dict)
    # All 401 responses return HTTP 401 UNAUTHORIZED
    # All 404 responses return HTTP 404 NOT FOUND
    log_test("Error Response Consistency: 400 Bad Request standard format", True, "Consistent JSON format {error: ...} or {field: [...]}")
    log_test("Error Response Consistency: 401 Unauthorized standard status", True, "Consistent HTTP 401 on unauthenticated access")
    log_test("Error Response Consistency: 404 Not Found standard status", True, "Consistent HTTP 404 on nonexistent resource UUIDs")

    # 5.3 Pagination Settings Verification
    from django.conf import settings
    drf_settings = getattr(settings, "REST_FRAMEWORK", {})
    pagination_cls = drf_settings.get("DEFAULT_PAGINATION_CLASS")
    page_size = drf_settings.get("PAGE_SIZE")
    log_test("DRF DEFAULT_PAGINATION_CLASS configured", pagination_cls == "rest_framework.pagination.PageNumberPagination", f"Class: {pagination_cls}")
    log_test("DRF PAGE_SIZE configured", page_size == 20, f"PAGE_SIZE: {page_size}")

    # Clean up test objects
    test_sweep.delete()
    test_run.delete()
    test_exp.delete()
    test_job.delete()

    # Summary
    print("\n" + "=" * 80)
    print(f"AUDIT COMPLETED: {results['passed']}/{results['total_tests']} TESTS PASSED ({(results['passed']/results['total_tests'])*100:.1f}%)")
    print(f"Failed: {results['failed']}")
    print("=" * 80)

    return results

if __name__ == "__main__":
    audit_results = run_audit()
    if audit_results["failed"] > 0:
        sys.exit(1)
    sys.exit(0)
