import uuid
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from rest_framework import status
from rest_framework.test import APIClient

from .models import CustomMetricEntry, Job, VersionedConfig


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------
def create_authenticated_client():
    user = User.objects.create_user(username="scheduser", password="schedpass")
    client = APIClient()
    client.force_authenticate(user=user)
    return client, user


@override_settings(
    ROOT_URLCONF="jobscheduler.test_urls"
)  # <-- override for the whole class
class JobschedulerTests(TestCase):
    def setUp(self):
        self.client, self.user = create_authenticated_client()
        self.list_create_url = "/api/jobs/"

        # Patch the scheduler globally to avoid thread startup and DB locks
        self.scheduler_patch = patch("jobscheduler.views.scheduler")
        self.mock_scheduler = self.scheduler_patch.start()
        self.addCleanup(self.scheduler_patch.stop)

        # Mock version-capture functions to avoid real git/dvc calls
        git_patcher = patch("jobscheduler.views.capture_git_info")
        dvc_patcher = patch("jobscheduler.views.capture_data_version")
        self.mock_git = git_patcher.start()
        self.mock_dvc = dvc_patcher.start()
        self.addCleanup(git_patcher.stop)
        self.addCleanup(dvc_patcher.stop)

    # ------------------------------- job_list GET / POST ------------------------------
    def test_list_jobs_empty(self):
        response = self.client.get(self.list_create_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data, [])

    def test_list_jobs_with_data(self):
        Job.objects.create(model_name="bert", task_name="mrpc")
        Job.objects.create(model_name="roberta", task_name="sst2")
        response = self.client.get(self.list_create_url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 2)

    def test_create_job_valid(self):
        payload = {
            "model_name": "distilbert",
            "task_name": "cola",
            "attention_type": "griffin_ssd",
            "epochs": 5,
            "batch_size": 32,
            "priority": 3,
        }
        response = self.client.post(self.list_create_url, payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        job = Job.objects.get(model_name="distilbert")
        self.assertEqual(job.task_name, "cola")
        self.assertEqual(job.attention_type, "griffin_ssd")
        self.mock_scheduler.add_job.assert_called_once()

    def test_create_job_missing_required(self):
        payload = {"model_name": "bert"}  # missing task_name
        response = self.client.post(self.list_create_url, payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    # ------------------------------- job_detail GET / PATCH / DELETE ------------------------------
    def create_job(self, **kwargs):
        return Job.objects.create(
            model_name=kwargs.pop("model_name", "bert"),
            task_name=kwargs.pop("task_name", "mrpc"),
            **kwargs,
        )

    def test_get_job_detail(self):
        job = self.create_job(name="Test Job")
        response = self.client.get(f"/api/jobs/{job.id}/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["name"], "Test Job")

    def test_patch_job_priority(self):
        job = self.create_job(priority=5, status="queued")
        response = self.client.patch(
            f"/api/jobs/{job.id}/", {"priority": 2}, format="json"
        )
        self.assertEqual(response.status_code, 200)
        job.refresh_from_db()
        self.assertEqual(job.priority, 2)
        self.mock_scheduler.update_priority.assert_called_with(str(job.id), 2)

    def test_patch_job_notes(self):
        job = self.create_job(notes="old note")
        response = self.client.patch(
            f"/api/jobs/{job.id}/", {"notes": "new note"}, format="json"
        )
        self.assertEqual(response.status_code, 200)
        job.refresh_from_db()
        self.assertEqual(job.notes, "new note")

    def test_delete_job_completed(self):
        job = self.create_job(status="completed")
        response = self.client.delete(f"/api/jobs/{job.id}/")
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertFalse(Job.objects.filter(id=job.id).exists())

    def test_delete_job_running_fails(self):
        job = self.create_job(status="running")
        response = self.client.delete(f"/api/jobs/{job.id}/")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    # ------------------------------- Scheduling Actions ------------------------------
    def test_pause_job(self):
        job = self.create_job(status="running")
        response = self.client.post(f"/api/jobs/{job.id}/pause/")
        self.assertEqual(response.status_code, 200)
        self.mock_scheduler.pause_job.assert_called_with(str(job.id))

    def test_resume_job(self):
        job = self.create_job(status="paused")
        response = self.client.post(f"/api/jobs/{job.id}/resume/")
        self.assertEqual(response.status_code, 200)
        self.mock_scheduler.resume_job.assert_called_with(str(job.id))

    def test_stop_job(self):
        job = self.create_job(status="running")
        response = self.client.post(f"/api/jobs/{job.id}/stop/")
        self.assertEqual(response.status_code, 200)
        self.mock_scheduler.stop_job.assert_called_with(str(job.id))

    def test_retry_failed_job(self):
        job = self.create_job(status="failed")
        response = self.client.post(f"/api/jobs/{job.id}/retry/")
        self.assertEqual(response.status_code, 200)
        job.refresh_from_db()
        self.assertEqual(job.status, "queued")
        self.mock_scheduler.add_job.assert_called_with(
            str(job.id), job.priority, job.estimated_duration, job.memory_requirements
        )

    def test_retry_completed_job_fails(self):
        job = self.create_job(status="completed")
        response = self.client.post(f"/api/jobs/{job.id}/retry/")
        self.assertEqual(response.status_code, 400)

    # ------------------------------- Training History ------------------------------
    def test_job_history(self):
        history = {
            "steps": [1, 2],
            "losses": [0.8, 0.5],
            "eval_steps": [1],
            "eval_losses": [0.7],
            "eval_metrics": [{"eval_accuracy": 0.6}],
        }
        job = self.create_job(training_history=history)
        response = self.client.get(f"/api/jobs/{job.id}/history/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["steps"], [1, 2])
        self.assertEqual(response.data["eval_accuracies"], [0.6])

    # ------------------------------- Job Notes ------------------------------
    def test_get_notes(self):
        job = self.create_job(notes="some note")
        response = self.client.get(f"/api/jobs/{job.id}/notes/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["notes"], "some note")

    def test_post_notes(self):
        job = self.create_job(notes="")
        response = self.client.post(
            f"/api/jobs/{job.id}/notes/", {"notes": "new note"}, format="json"
        )
        self.assertEqual(response.status_code, 200)
        job.refresh_from_db()
        self.assertEqual(job.notes, "new note")

    # ------------------------------- Configuration Versioning ------------------------------
    def test_config_history(self):
        job = self.create_job()
        VersionedConfig.objects.create(
            run=job,
            version_number=1,
            config_snapshot={"a": 1},
            change_description="Initial",
        )
        VersionedConfig.objects.create(
            run=job,
            version_number=2,
            config_snapshot={"a": 2},
            change_description="Update",
            previous_version=VersionedConfig.objects.first(),
        )
        response = self.client.get(f"/api/jobs/{job.id}/config/history/")
        self.assertEqual(response.status_code, 200)
        versions = response.data["versions"]
        self.assertEqual(len(versions), 2)
        self.assertEqual(versions[0]["version"], 2)

    def test_compare_configs(self):
        job = self.create_job()
        v1 = VersionedConfig.objects.create(
            run=job, version_number=1, config_snapshot={"lr": 1e-3, "batch": 16}
        )
        v2 = VersionedConfig.objects.create(
            run=job, version_number=2, config_snapshot={"lr": 2e-3, "batch": 32}
        )
        url = f"/api/jobs/{job.id}/config/compare/{v1.version_number}/{v2.version_number}/"
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertIn("values_changed", response.data)

    # ------------------------------- Custom Metrics ------------------------------
    def test_log_custom_metric(self):
        job = self.create_job()
        payload = {"name": "train_loss", "type": "scalar", "value": 0.5, "step": 10}
        response = self.client.post(
            f"/api/jobs/{job.id}/metrics/log/", payload, format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(CustomMetricEntry.objects.count(), 1)

    def test_get_custom_metrics(self):
        job = self.create_job()
        CustomMetricEntry.objects.create(
            run=job, name="loss", type="scalar", value=0.1, step=1
        )
        CustomMetricEntry.objects.create(
            run=job, name="loss", type="scalar", value=0.05, step=2
        )
        response = self.client.get(f"/api/jobs/{job.id}/metrics/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("loss", response.data)
        self.assertEqual(len(response.data["loss"]["steps"]), 2)

    # ------------------------------- Environment & Versioning ------------------------------
    def test_get_environment(self):
        job = self.create_job(
            python_version="3.10",
            pip_freeze="torch==2.0",
            git_commit_hash="abc123",
            git_branch="main",
            git_repository="https://github.com/org/repo",
            data_git_commit_id="def456",
            dvc_data_hash="ef789",
        )
        response = self.client.get(f"/api/jobs/{job.id}/environment/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["python_version"], "3.10")
        self.assertEqual(response.data["git"]["commit"], "abc123")

    # ------------------------------- System & GPU Info ------------------------------
    # @patch('jobscheduler.views.torch.cuda.is_available')
    # @patch('jobscheduler.views.torch.cuda.device_count')
    # @patch('jobscheduler.views.torch.cuda.get_device_properties')
    # @patch('jobscheduler.views.torch.cuda.memory_allocated')
    def test_gpu_info_cuda(self):
        import jobscheduler.views as jv

        mock_cuda = MagicMock()
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 2
        mock_cuda.get_device_properties.return_value.total_memory = 8 * 1024**3
        mock_cuda.memory_allocated.return_value = 1 * 1024**3
        mock_cuda.utilization.return_value = 50

        with patch.object(jv, "torch", MagicMock(cuda=mock_cuda)):
            Job.objects.create(
                model_name="m", task_name="t", status="running", gpu_requirements=1
            )
            response = self.client.get("/api/system/gpu/")
            self.assertEqual(response.status_code, 200)
            data = response.data
            self.assertEqual(data["total_gpus"], 2)
            self.assertEqual(data["available"], 1)
            self.assertEqual(data["used"], 1)

    def test_system_status(self):
        response = self.client.get("/api/system/status/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("cpu", response.data)

    def test_scheduler_status(self):
        self.mock_scheduler.total_gpus = 4
        self.mock_scheduler.available_gpus = 2
        self.mock_scheduler.task_queue.qsize.return_value = 3
        self.mock_scheduler.running_jobs = {"a": MagicMock()}

        response = self.client.get("/api/scheduler/status/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["total_gpus"], 4)
        self.assertEqual(response.data["queued_jobs"], 3)
        self.assertEqual(response.data["running_jobs"], 1)

    def test_scheduler_metrics(self):
        Job.objects.create(
            model_name="m",
            task_name="t",
            status="completed",
            estimated_duration=600,
            actual_duration=550,
            gpu_requirements=1,
            memory_requirements=8,
            priority=3,
        )
        response = self.client.get("/api/scheduler/metrics/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]["estimated_duration"], 600)

    # ================================================================
    # ADDITIONAL HARD TESTS (30)
    # ================================================================

    # 1. Create job with all optional fields
    def test_create_job_with_all_fields(self):
        payload = {
            "model_name": "roberta",
            "task_name": "sst2",
            "attention_type": "full",
            "method": "lora",
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 3e-5,
            "max_seq_length": 256,
            "compression_rank": 32,
            "gpu_requirements": 2,
            "memory_requirements": 16,
            "priority": 1,
            "estimated_duration": 3600,
            "notes": "High-priority experiment",
            "name": "Comprehensive Test",
        }
        resp = self.client.post(self.list_create_url, payload, format="json")
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        job = Job.objects.get(name="Comprehensive Test")
        self.assertEqual(job.gpu_requirements, 2)
        self.assertEqual(job.memory_requirements, 16)
        self.assertEqual(job.estimated_duration, 3600)

    # 2. Create job with extremely high priority (clamped to 10? view doesn't clamp on creation - only in patch)
    # But the view doesn't validate, it just stores whatever integer. So we test that any value is accepted.
    def test_create_job_extreme_priority(self):
        payload = {"model_name": "bert", "task_name": "mrpc", "priority": 999}
        resp = self.client.post(self.list_create_url, payload, format="json")
        self.assertEqual(resp.status_code, 201)
        job = Job.objects.first()
        self.assertEqual(job.priority, 999)

    # 3. Create job with negative priority (allowed by model, not validated)
    def test_create_job_negative_priority(self):
        payload = {"model_name": "bert", "task_name": "mrpc", "priority": -5}
        resp = self.client.post(self.list_create_url, payload, format="json")
        self.assertEqual(resp.status_code, 201)
        job = Job.objects.first()
        self.assertEqual(job.priority, -5)

    # 4. Patch job with a non‑integer priority should fail gracefully? The view tries to convert to int, but DRF may accept string. We'll test with string.
    # In jobscheduler/tests.py
    def test_patch_job_priority_invalid_type(self):
        job = self.create_job(priority=5)
        resp = self.client.patch(
            f"/api/jobs/{job.id}/", {"priority": "high"}, format="json"
        )
        self.assertEqual(
            resp.status_code, 400
        )  # not 500        # The view will try max(1, min(10, 'high')) → TypeError. So 500 is expected.

    # 5. Patch job with priority 0 (will be clamped to 1 by view)
    def test_patch_job_priority_clamped(self):
        job = self.create_job(priority=5)
        resp = self.client.patch(f"/api/jobs/{job.id}/", {"priority": 0}, format="json")
        self.assertEqual(resp.status_code, 200)
        job.refresh_from_db()
        self.assertEqual(job.priority, 1)  # 0 → max(1,0)=1

    # 6. Delete a job that is queued (should be prevented)
    def test_delete_queued_job(self):
        job = self.create_job(status="queued")
        resp = self.client.delete(f"/api/jobs/{job.id}/")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertTrue(Job.objects.filter(id=job.id).exists())

    # 7. Pause a job that doesn't exist (404)
    def test_pause_nonexistent_job(self):
        resp = self.client.post(f"/api/jobs/{uuid.uuid4()}/pause/")
        self.assertEqual(resp.status_code, 404)

    # 8. Resume a job that is not paused (should succeed but scheduler may complain; test it’s callable)
    def test_resume_non_paused_job(self):
        job = self.create_job(status="running")
        resp = self.client.post(f"/api/jobs/{job.id}/resume/")
        self.assertEqual(resp.status_code, 200)

    # 9. Stop a job that is already completed (the view just calls scheduler.stop_job regardless – scheduler mock handles it)
    def test_stop_completed_job(self):
        job = self.create_job(status="completed")
        resp = self.client.post(f"/api/jobs/{job.id}/stop/")
        self.assertEqual(resp.status_code, 200)

    # 10. Retry a job that is currently running (should fail)
    def test_retry_running_job(self):
        job = self.create_job(status="running")
        resp = self.client.post(f"/api/jobs/{job.id}/retry/")
        self.assertEqual(resp.status_code, 400)

    # 11. Get notes for a job with no notes (empty)
    def test_get_notes_empty(self):
        job = self.create_job(notes="")
        resp = self.client.get(f"/api/jobs/{job.id}/notes/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["notes"], "")

    # 12. Post notes with empty string (should clear notes)
    def test_post_notes_clear(self):
        job = self.create_job(notes="existing")
        resp = self.client.post(
            f"/api/jobs/{job.id}/notes/", {"notes": ""}, format="json"
        )
        self.assertEqual(resp.status_code, 200)
        job.refresh_from_db()
        self.assertEqual(job.notes, "")

    # 13. Config history for a job with no versions (should return empty list)
    def test_config_history_empty(self):
        job = self.create_job()
        resp = self.client.get(f"/api/jobs/{job.id}/config/history/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["versions"], [])

    # 14. Compare configs with non‑existent version (404)
    def test_compare_configs_nonexistent_version(self):
        job = self.create_job()
        VersionedConfig.objects.create(run=job, version_number=1, config_snapshot={})
        resp = self.client.get(f"/api/jobs/{job.id}/config/compare/1/99/")
        self.assertEqual(resp.status_code, 404)

    # 15. Log custom metric with missing required field 'step' (should be 400)
    def test_log_custom_metric_missing_field(self):
        job = self.create_job()
        payload = {"name": "loss", "type": "scalar", "value": 0.5}  # missing step
        resp = self.client.post(
            f"/api/jobs/{job.id}/metrics/log/", payload, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    # 16. Log custom metric with invalid type (should be 400)
    def test_log_custom_metric_invalid_type(self):
        job = self.create_job()
        payload = {"name": "loss", "type": "invalid", "value": 0.5, "step": 1}
        resp = self.client.post(
            f"/api/jobs/{job.id}/metrics/log/", payload, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    # 17. Get custom metrics for a job with none (should return empty dict)
    def test_get_custom_metrics_empty(self):
        job = self.create_job()
        resp = self.client.get(f"/api/jobs/{job.id}/metrics/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data, {})

    # 18. Environment for a job with minimal info (all fields None)
    def test_get_environment_minimal(self):
        job = self.create_job()
        resp = self.client.get(f"/api/jobs/{job.id}/environment/")
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.data["python_version"])
        self.assertIsNone(resp.data["git"]["commit"])

    # 19. Update job scheduling – change priority of a running job (should update DB but not scheduler since job is running)
    def test_update_scheduling_priority_running(self):
        job = self.create_job(status="running", priority=5)
        resp = self.client.patch(
            f"/api/jobs/{job.id}/scheduling/", {"priority": 2}, format="json"
        )
        self.assertEqual(resp.status_code, 200)
        job.refresh_from_db()
        self.assertEqual(job.priority, 2)
        # Scheduler.update_priority should NOT be called because job is not queued
        self.mock_scheduler.update_priority.assert_not_called()

    # 20. Update job scheduling – change priority of a queued job (should call scheduler)
    def test_update_scheduling_priority_queued(self):
        job = self.create_job(status="queued", priority=5)
        resp = self.client.patch(
            f"/api/jobs/{job.id}/scheduling/", {"priority": 2}, format="json"
        )
        self.assertEqual(resp.status_code, 200)
        self.mock_scheduler.update_priority.assert_called_with(str(job.id), 2)

    # 21. Update job scheduling – change estimated_duration
    def test_update_scheduling_duration(self):
        job = self.create_job()
        resp = self.client.patch(
            f"/api/jobs/{job.id}/scheduling/",
            {"estimated_duration": 7200},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        job.refresh_from_db()
        self.assertEqual(job.estimated_duration, 7200)

    # 22. Job stats – creates jobs with different statuses and verifies counts
    def test_job_stats(self):
        self.create_job(status="queued", name="q")
        self.create_job(status="running", name="r")
        self.create_job(status="completed", name="c", actual_duration=300)
        resp = self.client.get("/api/jobs/stats/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["total"], 3)
        self.assertEqual(resp.data["by_status"]["queued"], 1)
        self.assertEqual(resp.data["by_status"]["running"], 1)
        self.assertEqual(resp.data["by_status"]["completed"], 1)
        self.assertAlmostEqual(resp.data["average_completion_seconds"], 300.0)

    # 23. Job stats with no completed jobs (avg should be None)
    def test_job_stats_no_completed(self):
        self.create_job(status="running")
        resp = self.client.get("/api/jobs/stats/")
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.data["average_completion_seconds"])

    # 24. Bulk delete – delete multiple completed jobs
    def test_bulk_delete(self):
        j1 = self.create_job(status="completed")
        j2 = self.create_job(status="failed")
        resp = self.client.post(
            "/api/jobs/bulk-delete/",
            {"job_ids": [str(j1.id), str(j2.id)]},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["deleted"], 2)
        self.assertFalse(Job.objects.filter(id__in=[j1.id, j2.id]).exists())

    # 25. Bulk delete – skip running/queued jobs
    def test_bulk_delete_skip_running(self):
        j1 = self.create_job(status="running")
        j2 = self.create_job(status="completed")
        resp = self.client.post(
            "/api/jobs/bulk-delete/",
            {"job_ids": [str(j1.id), str(j2.id)]},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["deleted"], 1)
        self.assertTrue(Job.objects.filter(id=j1.id).exists())
        self.assertFalse(Job.objects.filter(id=j2.id).exists())

    # 26. Bulk delete – missing job_ids
    def test_bulk_delete_missing_ids(self):
        resp = self.client.post("/api/jobs/bulk-delete/", {}, format="json")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.data)

    # 27. Job detail with invalid UUID format (should be 404 automatically)
    def test_job_detail_invalid_uuid(self):
        resp = self.client.get("/api/jobs/not-a-uuid/")
        self.assertEqual(resp.status_code, 404)

    # 28. List jobs with pagination? The view doesn't paginate, so just check ordering
    def test_list_jobs_ordering(self):
        Job.objects.create(model_name="a", task_name="a", priority=1, name="first")
        Job.objects.create(model_name="b", task_name="b", priority=2, name="second")
        resp = self.client.get(self.list_create_url)
        self.assertEqual(
            resp.data[0]["name"], "first"
        )  # ordered by priority asc, then -created_at

    # 29. System status endpoint (already tested for 200, also check presence of memory keys)
    def test_system_status_structure(self):
        resp = self.client.get("/api/system/status/")
        self.assertIn("cpu", resp.data)
        self.assertIn("memory", resp.data)
        self.assertIn("memory_used_gb", resp.data)

    # 30. Scheduler metrics with no completed jobs (should return empty list)
    def test_scheduler_metrics_empty(self):
        resp = self.client.get("/api/scheduler/metrics/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data, [])


from celery import shared_task

from .predictive import PriorityPredictor


@shared_task
def retrain_priority_model():
    predictor = PriorityPredictor()
    predictor.train()
    return {"status": "retrained", "model_path": predictor.model_path}
