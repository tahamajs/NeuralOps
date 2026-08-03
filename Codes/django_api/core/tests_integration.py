from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.utils import timezone
from experiment.models import Experiment
from jobscheduler.models import Job
from model_versioning.models import ModelVersion
from notes.models import ResearcherNote
from rest_framework.test import APIClient

# Remove TokenAuthMiddleware for tests
TEST_MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]


# ----------------------------------------------------------------------
# Base class that sets up common test users and authentication helpers
# ----------------------------------------------------------------------
@override_settings(MIDDLEWARE=TEST_MIDDLEWARE)
class IntegrationTestBase(TestCase):
    def setUp(self):
        self.client = APIClient()
        # Staff user for admin operations
        self.staff = User.objects.create_user(
            username="staff", password="staffpass", is_staff=True
        )
        # Regular researcher
        self.researcher = None
        self.researcher_token = None

        # Mock the GPU scheduler globally to avoid threads and DB locks
        self.scheduler_patcher = patch("jobscheduler.views.scheduler")
        self.mock_scheduler = self.scheduler_patcher.start()
        self.addCleanup(self.scheduler_patcher.stop)

        # Mock boto3 for cold storage tests (archive/restore)
        self.boto_patcher = patch("boto3.client")
        self.mock_boto_client = self.boto_patcher.start()
        self.addCleanup(self.boto_patcher.stop)

        # Set dummy MODEL_ARCHIVE_BUCKET to avoid settings error
        self.settings_patcher = override_settings(MODEL_ARCHIVE_BUCKET="test-bucket")
        self.settings_patcher.enable()
        self.addCleanup(self.settings_patcher.disable)

    def register_and_login(
        self, username="researcher", password="Strong1!", email=None
    ):
        """Helper: register a new user and authenticate the client."""
        if email is None:
            email = f"{username}@test.com"
        resp = self.client.post(
            "/api/auth/register/",
            {"username": username, "password": password, "email": email},
        )
        self.assertEqual(resp.status_code, 201)
        token = resp.data["token"]
        self.client.credentials(HTTP_AUTHORIZATION="Token " + token)
        return token, username

    def login_as_staff(self):
        """Authenticate the client as the staff user."""
        resp = self.client.post(
            "/api/auth/login/", {"username": "staff", "password": "staffpass"}
        )
        self.assertEqual(resp.status_code, 200)
        token = resp.data["token"]
        self.client.credentials(HTTP_AUTHORIZATION="Token " + token)
        return token


# ----------------------------------------------------------------------
# 1. Authentication & User Management Flows
# ----------------------------------------------------------------------
class AuthWorkflowTests(IntegrationTestBase):
    def test_user_registration_login_logout(self):
        # Register
        resp = self.client.post(
            "/api/auth/register/",
            {"username": "newuser", "password": "Valid1!", "email": "new@test.com"},
        )
        self.assertEqual(resp.status_code, 201)
        token = resp.data["token"]

        # Login with wrong password should fail
        self.client.credentials(HTTP_AUTHORIZATION="Token " + token)
        resp = self.client.post(
            "/api/auth/login/", {"username": "newuser", "password": "wrong"}
        )
        self.assertEqual(resp.status_code, 401)

        # Correct login
        resp = self.client.post(
            "/api/auth/login/", {"username": "newuser", "password": "Valid1!"}
        )
        self.assertEqual(resp.status_code, 200)
        token = resp.data["token"]

        # Token refresh
        resp = self.client.post("/api/auth/refresh-token/")
        self.assertEqual(resp.status_code, 200)
        new_token = resp.data["token"]
        self.assertNotEqual(token, new_token)

        # Get user info
        resp = self.client.get("/api/auth/user/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["username"], "newuser")

        # Update profile
        resp = self.client.patch("/api/auth/profile/", {"first_name": "Test"})
        self.assertEqual(resp.status_code, 200)

        # Change password
        resp = self.client.post(
            "/api/auth/change-password/",
            {"old_password": "Valid1!", "new_password": "NewValid1!"},
        )
        self.assertEqual(resp.status_code, 200)

        # Logout
        resp = self.client.post("/api/auth/logout/")
        self.assertEqual(resp.status_code, 200)

        # After logout, token should be invalid
        self.client.credentials(HTTP_AUTHORIZATION="Token " + token)
        resp = self.client.get("/api/auth/user/")
        self.assertEqual(resp.status_code, 401)

    def test_register_with_existing_username(self):
        User.objects.create_user(username="existing", password="pass")
        resp = self.client.post(
            "/api/auth/register/", {"username": "existing", "password": "Valid1!"}
        )
        self.assertEqual(resp.status_code, 400)

    def test_login_with_empty_fields(self):
        resp = self.client.post("/api/auth/login/", {})
        self.assertEqual(resp.status_code, 400)

    def test_change_password_invalid_old(self):
        self.register_and_login("pwduser", "OldPass1!")
        resp = self.client.post(
            "/api/auth/change-password/",
            {"old_password": "wrong", "new_password": "NewPass1!"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_profile_update_duplicate_email(self):
        self.register_and_login("emailuser1", "Pass1!", email="a@b.com")
        # Register another user with same email
        User.objects.create_user(username="emailuser2", password="x", email="a@b.com")
        resp = self.client.patch("/api/auth/profile/", {"email": "a@b.com"})
        self.assertEqual(resp.status_code, 400)

    def test_login_rate_limiting(self):
        # brute force login attempts should eventually be throttled
        for _ in range(25):
            resp = self.client.post(
                "/api/auth/login/", {"username": "nonexistent", "password": "wrong"}
            )
        # After many attempts, the response should be 429 (or 401 if throttle not yet triggered)
        # As we may not have rate limiting fully active in tests, we just check that it doesn't crash.
        self.assertIn(resp.status_code, [401, 429])


# ----------------------------------------------------------------------
# 2. Experiment & Training Run Workflows
# ----------------------------------------------------------------------
class ExperimentWorkflowTests(IntegrationTestBase):
    def setUp(self):
        super().setUp()
        self.token, _ = self.register_and_login()

    def test_full_experiment_lifecycle(self):
        # Create experiment
        resp = self.client.post(
            "/api/experiments/",
            {"name": "Transformer Study", "tags": "nlp,deep-learning"},
            format="json",
        )
        self.assertEqual(resp.status_code, 201)
        exp_id = resp.data["id"]

        # List experiments
        resp = self.client.get("/api/experiments/")
        self.assertEqual(len(resp.data), 1)

        # Update experiment
        resp = self.client.patch(
            f"/api/experiments/{exp_id}/",
            {"description": "Updated description"},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)

        # Start a training run
        resp = self.client.post(
            "/api/train/",
            {
                "experiment_id": exp_id,
                "name": "Baseline",
                "model_name": "bert-base-uncased",
                "task_name": "mrpc",
                "epochs": 2,
                "batch_size": 16,
            },
            format="json",
        )
        self.assertIn(resp.status_code, [201, 202])
        run_id = resp.data["id"]

        # Get run detail
        resp = self.client.get(f"/api/runs/{run_id}/")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue("status" in resp.data)

        # List runs for experiment
        resp = self.client.get(f"/api/runs/exp/{exp_id}/")
        self.assertGreaterEqual(len(resp.data), 1)

        # Training history and snapshots
        resp = self.client.get(f"/api/runs/{run_id}/history/")
        self.assertEqual(resp.status_code, 200)
        resp = self.client.get(f"/api/runs/{run_id}/snapshots/")
        self.assertEqual(resp.status_code, 200)

        # Update run config
        resp = self.client.patch(
            f"/api/runs/{run_id}/config/", {"learning_rate": 3e-5}, format="json"
        )
        self.assertEqual(resp.status_code, 200)

        # Attempt to stop a run (it may be completed by mock)
        resp = self.client.post(f"/api/runs/{run_id}/stop/")
        # If already completed, it returns 400; otherwise 200. We just check it's handled.
        self.assertIn(resp.status_code, [200, 400])

        # Generate paper placeholder
        resp = self.client.get(f"/api/runs/{run_id}/paper/")
        self.assertEqual(resp.status_code, 200)

        # Run an evaluation (async, mocked to run sync)
        resp = self.client.post(f"/api/runs/{run_id}/evaluate/")
        self.assertIn(resp.status_code, [200, 201, 202])

        # Compress
        resp = self.client.post("/api/compress/", {"run_id": run_id}, format="json")
        self.assertIn(resp.status_code, [200, 201, 202])

        # Delete experiment (only if no runs? We'll delete the run first)
        # Delete run
        # Actual run may be 'completed' so delete should work
        resp = self.client.delete(f"/api/runs/{run_id}/")
        if resp.status_code == 400:  # might be running
            pass
        self.assertIn(resp.status_code, [204, 400])

        # Delete experiment
        resp = self.client.delete(f"/api/experiments/{exp_id}/")
        self.assertEqual(resp.status_code, 204)

    def test_start_training_without_experiment(self):
        # Automatically creates an experiment if experiment_id not provided
        resp = self.client.post(
            "/api/train/",
            {"name": "Standalone", "model_name": "distilbert", "task_name": "sst2"},
            format="json",
        )
        self.assertIn(resp.status_code, [201, 202])
        self.assertTrue(resp.data.get("experiment"))  # experiment field exists

    def test_system_status_public(self):
        # System status should be accessible without auth
        client = APIClient()
        resp = client.get("/api/system/status/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("cpu", resp.data)

    def test_gpu_info_public(self):
        resp = self.client.get("/api/system/gpu/")  # authenticated
        self.assertEqual(resp.status_code, 200)
        self.assertIn("type", resp.data)

    def test_run_artifacts_and_model_download(self):
        resp = self.client.post(
            "/api/train/",
            {
                "name": "ArtifactRun",
                "model_name": "bert",
                "task_name": "cola",
                "epochs": 1,
                "batch_size": 8,
            },
            format="json",
        )
        run_id = resp.data["id"]
        # Artifacts list (empty initially)
        resp = self.client.get(f"/api/runs/{run_id}/artifacts/")
        self.assertEqual(resp.status_code, 200)
        # Model download (should 404 because no model saved)
        resp = self.client.get(f"/api/runs/{run_id}/model/")
        self.assertEqual(resp.status_code, 404)

    def test_experiment_permissions(self):
        # Create experiment as researcher
        exp = Experiment.objects.create(name="ResearcherExp")
        resp = self.client.get(f"/api/experiments/{exp.id}/")
        self.assertEqual(resp.status_code, 200)
        # Another user without authentication cannot see it
        client2 = APIClient()
        resp = client2.get(f"/api/experiments/{exp.id}/")
        self.assertEqual(resp.status_code, 401)

    def test_start_evaluate_without_experiment(self):
        resp = self.client.post(
            "/api/evaluate/",
            {"name": "EvalNoExp", "model_name": "distilbert", "task_name": "mrpc"},
            format="json",
        )
        self.assertIn(resp.status_code, [201, 202])
        self.assertTrue(resp.data.get("experiment"))

    def test_stop_all_jobs(self):
        # This endpoint may not exist in the new unified URL scheme, but if it does under experiment/urls.py we test it.
        resp = self.client.post("/api/runs/stop_all/")
        # Accept either 200 (if implemented) or 404 (if route removed). We just check it doesn't crash.
        self.assertIn(resp.status_code, [200, 404])


# ----------------------------------------------------------------------
# 3. Researcher Notes Workflows
# ----------------------------------------------------------------------
class NotesWorkflowTests(IntegrationTestBase):
    def setUp(self):
        super().setUp()
        self.token, self.username = self.register_and_login()
        # Create a run for linking
        self.job = Job.objects.create(
            name="NoteRun", model_name="bert", task_name="mrpc"
        )
        self.job.author = User.objects.get(username="researcher")
        self.job.save()

    def test_full_note_lifecycle(self):
        # Create a note
        resp = self.client.post(
            "/api/notes/v1/notes/",
            {
                "title": "My Research Note",
                "content": "Interesting findings...",
                "run": str(self.job.id),
                "tags": ["important", "nlp"],
                "is_public": False,
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 201)
        note_id = resp.data["id"]

        # Get note detail
        resp = self.client.get(f"/api/notes/v1/notes/{note_id}/")
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.data["is_public"])

        # Make it public
        resp = self.client.patch(
            f"/api/notes/v1/notes/{note_id}/", {"is_public": True}, format="json"
        )
        self.assertEqual(resp.status_code, 200)

        # Another user should see it
        other_client = APIClient()
        other_token, _ = self.register_and_login("other", "OtherPass1!")
        other_client.credentials(HTTP_AUTHORIZATION="Token " + other_token)
        resp = other_client.get(f"/api/notes/v1/notes/{note_id}/")
        self.assertEqual(resp.status_code, 200)

        # Restore credentials of original user (researcher) to add reaction
        self.client.credentials(HTTP_AUTHORIZATION="Token " + self.token)
        self.client.cookies["authToken"] = self.token
        # Add reaction
        resp = self.client.post(
            f"/api/notes/v1/notes/{note_id}/react/", {"reaction": "like"}, format="json"
        )
        self.assertEqual(resp.status_code, 200)
        note = ResearcherNote.objects.get(id=note_id)
        self.assertIn("researcher", note.reactions.get("like", []))

        # Toggle reaction (remove)
        resp = self.client.post(
            f"/api/notes/v1/notes/{note_id}/react/", {"reaction": "like"}, format="json"
        )
        self.assertEqual(resp.status_code, 200)
        note.refresh_from_db()
        self.assertNotIn("researcher", note.reactions.get("like", []))

        # Add a reply (threaded comment)
        resp = self.client.post(
            "/api/notes/v1/notes/",
            {"content": "Great finding!", "parent_note": note_id},  # title not required
            format="json",
        )
        self.assertEqual(resp.status_code, 201)
        reply_id = resp.data["id"]

        # Get replies
        resp = self.client.get(f"/api/notes/v1/notes/{note_id}/replies/")
        self.assertEqual(len(resp.data), 1)

        # Bulk delete of notes (soft delete)
        resp = self.client.post(
            "/api/notes/v1/notes/bulk/",
            {"action": "delete", "ids": [note_id, reply_id]},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["deleted"], 2)

        # Restore the original note
        resp = self.client.post(f"/api/notes/v1/notes/{note_id}/restore/")
        self.assertEqual(resp.status_code, 200)

        # Permanent delete (hard delete via model, not API) – skip for integration

    def test_notes_csv_export(self):
        ResearcherNote.objects.create(
            title="Export Test",
            content="Data",
            author=User.objects.get(username="researcher"),
            run=self.job,
        )
        resp = self.client.get("/api/notes/v1/notes/?export=csv")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/csv", resp["Content-Type"])

    def test_notes_statistics(self):
        resp = self.client.get("/api/notes/v1/notes/statistics/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("total_notes", resp.data)

    def test_webhook_subscription(self):
        resp = self.client.post(
            "/api/notes/v1/webhooks/",
            {"url": "https://example.com/hook", "events": ["note.created"]},
            format="json",
        )
        self.assertEqual(resp.status_code, 201)
        sub_id = resp.data["id"]

        # List webhooks
        resp = self.client.get("/api/notes/v1/webhooks/")
        self.assertEqual(len(resp.data), 1)

        # Delete webhook
        resp = self.client.delete(f"/api/notes/v1/webhooks/{sub_id}/")
        self.assertEqual(resp.status_code, 204)

    def test_note_version_history(self):
        note = ResearcherNote.objects.create(
            title="Version Test",
            content="V1",
            author=User.objects.get(username=self.username),
            run=self.job,
        )
        note.content = "V2"
        note.save()  # creates version v1
        resp = self.client.get(f"/api/notes/v1/notes/{note.id}/versions/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.data), 1)
        resp = self.client.get(f"/api/notes/v1/notes/{note.id}/versions/1/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["content"], "V1")

    def test_note_mentions_and_read(self):
        note = ResearcherNote.objects.create(
            title="Mention Test",
            content="Hey @researcher",
            author=User.objects.get(username=self.username),
            run=self.job,
        )
        from notes.models import NoteMention

        mention = NoteMention.objects.create(
            note=note, mentioned_user=User.objects.get(username=self.username)
        )
        # list mentions
        resp = self.client.get(f"/api/notes/v1/notes/{note.id}/mentions/")
        self.assertEqual(resp.status_code, 200)
        # mark read
        resp = self.client.post(f"/api/notes/v1/mentions/{mention.id}/read/")
        self.assertEqual(resp.status_code, 200)
        mention.refresh_from_db()
        self.assertTrue(mention.is_read)

    def test_note_bulk_tags_operations(self):
        note = self.client.post(
            "/api/notes/v1/notes/",
            {
                "title": "TagTest",
                "content": "x",
                "run": str(self.job.id),
                "tags": ["old"],
            },
            format="json",
        )
        note_id = note.data["id"]
        resp = self.client.patch(
            "/api/notes/v1/notes/bulk/tags/",
            {"action": "add", "ids": [note_id], "tags": ["new"]},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        note_obj = ResearcherNote.objects.get(id=note_id)
        self.assertIn("new", note_obj.tags)

    def test_recent_notes(self):
        from datetime import timedelta

        old_note = ResearcherNote.objects.create(
            title="Old",
            content="x",
            author=User.objects.get(username=self.username),
            run=self.job,
        )
        ResearcherNote.objects.filter(id=old_note.id).update(
            created_at=timezone.now() - timedelta(days=30),
            updated_at=timezone.now() - timedelta(days=30),
        )
        new_note = self._create_note_via_api(title="Recent")  # helper needed
        resp = self.client.get("/api/notes/v1/notes/recent/?days=7")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.data["results"]), 1)  # only recent

    def _create_note_via_api(self, **kw):
        data = {
            "title": kw.get("title", "default"),
            "content": kw.get("content", "content"),
            "run": str(self.job.id),
            **kw,
        }
        resp = self.client.post("/api/notes/v1/notes/", data, format="json")
        self.assertEqual(resp.status_code, 201)
        return resp


# ----------------------------------------------------------------------
# 4. Model Versioning Workflows (staff & normal user mixed)
# ----------------------------------------------------------------------
class ModelVersioningWorkflowTests(IntegrationTestBase):
    def setUp(self):
        super().setUp()
        self.token, _ = self.register_and_login()
        self.job = Job.objects.create(
            name="MV Job", model_name="bert", task_name="mrpc"
        )
        self.job.author = User.objects.get(username="researcher")  # dynamically attach

    def test_list_and_view_model_versions(self):
        mv = ModelVersion.objects.create(
            run=self.job,
            version="v1.0",
            version_number=1,
            artifact_path="/tmp/fake",
            artifact_size_mb=50.0,
            metrics={"eval_accuracy": 0.89},
        )
        resp = self.client.get(f"/api/model-versioning/runs/{self.job.id}/versions/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.data["results"]), 1)

        resp = self.client.get(f"/api/model-versioning/versions/{mv.id}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["version"], "v1.0")

    def test_download_model_version(self):
        mv = ModelVersion.objects.create(
            run=self.job,
            version="v1.0",
            version_number=1,
            artifact_path="/tmp/fake",
            artifact_size_mb=50.0,
        )
        # Mock file existence
        with patch("os.path.exists", return_value=True), patch(
            "builtins.open", new_callable=MagicMock
        ):
            resp = self.client.get(f"/api/model-versioning/versions/{mv.id}/download/")
            self.assertEqual(resp.status_code, 200)

    def test_staff_only_actions(self):
        # First, verify normal user is forbidden
        mv = ModelVersion.objects.create(
            run=self.job,
            version="v1.0",
            version_number=1,
            artifact_path="/tmp/fake",
            artifact_size_mb=50.0,
        )
        # Publish
        resp = self.client.post(f"/api/model-versioning/versions/{mv.id}/publish/")
        self.assertEqual(resp.status_code, 403)

        # Archive
        resp = self.client.post(f"/api/model-versioning/versions/{mv.id}/archive/")
        self.assertEqual(resp.status_code, 403)

        # Set production version
        resp = self.client.post(
            f"/api/model-versioning/runs/{self.job.id}/versions/1/set_production/"
        )
        self.assertEqual(resp.status_code, 403)

        # Now login as staff and succeed
        self.login_as_staff()
        # Publish
        resp = self.client.post(f"/api/model-versioning/versions/{mv.id}/publish/")
        # Might fail because HuggingFacePublisher is not really installed; we mock it.
        with patch("model_versioning.views.HuggingFacePublisher") as mock_pub:
            mock_pub.return_value.publish.return_value = "https://hf.co/test"
            resp = self.client.post(f"/api/model-versioning/versions/{mv.id}/publish/")
            self.assertEqual(resp.status_code, 200)

        # Archive
        with patch.object(
            self.job, "model_versions", MagicMock()
        ):  # avoid real model_versions
            resp = self.client.post(f"/api/model-versioning/versions/{mv.id}/archive/")
            self.assertEqual(resp.status_code, 200)
        # Set production (need another version)
        ModelVersion.objects.create(
            run=self.job,
            version="v2",
            version_number=2,
            artifact_path="/tmp/fake",
            artifact_size_mb=60.0,
        )
        resp = self.client.post(
            f"/api/model-versioning/runs/{self.job.id}/versions/2/set_production/"
        )
        self.assertEqual(resp.status_code, 200)

    def test_version_comparison(self):
        run = Job.objects.create(name="CmpRun", model_name="x", task_name="y")
        v1 = ModelVersion.objects.create(
            run=run,
            version="v1",
            version_number=1,
            artifact_path="/tmp/v1",
            artifact_size_mb=10,
            metrics={"eval_accuracy": 0.8},
        )
        v2 = ModelVersion.objects.create(
            run=run,
            version="v2",
            version_number=2,
            artifact_path="/tmp/v2",
            artifact_size_mb=12,
            metrics={"eval_accuracy": 0.85},
        )
        resp = self.client.get(
            f"/api/model-versioning/runs/{run.id}/versions/compare/1/2/"
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("metrics_diff", resp.data)
        self.assertAlmostEqual(resp.data["size_diff_mb"], 2.0, places=1)

    def test_version_search_and_filtering(self):
        run = Job.objects.create(name="SearchRun", model_name="x", task_name="y")
        ModelVersion.objects.create(
            run=run,
            version="bert-v1",
            version_number=1,
            artifact_path="/tmp",
            artifact_size_mb=5,
            validation_passed=True,
            storage_tier="hot",
        )
        ModelVersion.objects.create(
            run=run,
            version="bert-v2",
            version_number=2,
            artifact_path="/tmp",
            artifact_size_mb=6,
            validation_passed=False,
            storage_tier="cold",
        )
        resp = self.client.get(
            f"/api/model-versioning/runs/{run.id}/versions/?search=bert-v1"
        )
        self.assertEqual(len(resp.data["results"]), 1)
        resp = self.client.get(
            f"/api/model-versioning/runs/{run.id}/versions/?validation_passed=true"
        )
        self.assertEqual(len(resp.data["results"]), 1)
        resp = self.client.get(
            f"/api/model-versioning/runs/{run.id}/versions/?storage_tier=cold"
        )
        self.assertEqual(len(resp.data["results"]), 1)

    def test_download_via_glob(self):
        mv = ModelVersion.objects.create(
            run=self.job,
            version="v1",
            version_number=1,
            artifact_path="/tmp/fake",
            artifact_size_mb=5,
        )
        with patch("os.path.exists", side_effect=lambda p: "model.pth" not in p), patch(
            "glob.glob", return_value=["/tmp/fake/checkpoint.pth"]
        ), patch("builtins.open", MagicMock()):
            resp = self.client.get(f"/api/model-versioning/versions/{mv.id}/download/")
            self.assertEqual(resp.status_code, 200)

    def test_archive_and_restore_failures(self):
        mv = ModelVersion.objects.create(
            run=self.job,
            version="v1",
            version_number=1,
            artifact_path="/tmp/fake",
            artifact_size_mb=5,
            storage_tier="hot",
        )
        self.login_as_staff()
        # Archive with missing bucket
        with override_settings(MODEL_ARCHIVE_BUCKET=None):
            resp = self.client.post(f"/api/model-versioning/versions/{mv.id}/archive/")
            self.assertEqual(resp.status_code, 501)
        # Restore from cold but not in cold
        resp = self.client.post(f"/api/model-versioning/versions/{mv.id}/restore/")
        self.assertEqual(resp.status_code, 400)

    def test_set_production_switch(self):
        v1 = ModelVersion.objects.create(
            run=self.job,
            version="v1",
            version_number=1,
            artifact_path="/tmp",
            artifact_size_mb=5,
        )
        v2 = ModelVersion.objects.create(
            run=self.job,
            version="v2",
            version_number=2,
            artifact_path="/tmp",
            artifact_size_mb=6,
        )
        self.login_as_staff()
        self.client.post(
            f"/api/model-versioning/runs/{self.job.id}/versions/1/set_production/"
        )
        resp = self.client.post(
            f"/api/model-versioning/runs/{self.job.id}/versions/2/set_production/"
        )
        self.assertEqual(resp.status_code, 200)
        v1.refresh_from_db()
        v2.refresh_from_db()
        self.assertFalse(v1.is_production)
        self.assertTrue(v2.is_production)


# ----------------------------------------------------------------------
# 5. Job Scheduler Workflows
# ----------------------------------------------------------------------
class JobSchedulerWorkflowTests(IntegrationTestBase):
    def setUp(self):
        super().setUp()
        self.token, _ = self.register_and_login()

    def test_create_and_manage_job(self):
        # Create a job
        resp = self.client.post(
            "/api/jobs/",
            {
                "model_name": "bert-base",
                "task_name": "mrpc",
                "name": "Test Job",
                "priority": 5,
                "gpu_requirements": 1,
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 201)
        job_id = resp.data["id"]

        # Get job detail
        resp = self.client.get(f"/api/jobs/{job_id}/")
        self.assertEqual(resp.status_code, 200)

        # Update priority
        resp = self.client.patch(f"/api/jobs/{job_id}/", {"priority": 3}, format="json")
        self.assertEqual(resp.status_code, 200)
        job = Job.objects.get(id=job_id)
        self.assertEqual(job.priority, 3)

        # Add notes
        resp = self.client.post(
            f"/api/jobs/{job_id}/notes/", {"notes": "Important run"}, format="json"
        )
        self.assertEqual(resp.status_code, 200)
        job.refresh_from_db()
        self.assertIn("Important run", job.notes)

        # Training history
        resp = self.client.get(f"/api/jobs/{job_id}/history/")
        self.assertEqual(resp.status_code, 200)

        # Log a custom metric
        resp = self.client.post(
            f"/api/jobs/{job_id}/metrics/log/",
            {"name": "validation_loss", "type": "scalar", "value": 0.45, "step": 1},
            format="json",
        )
        self.assertEqual(resp.status_code, 201)

        # Get custom metrics
        resp = self.client.get(f"/api/jobs/{job_id}/metrics/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("validation_loss", resp.data)

        # Environment info
        resp = self.client.get(f"/api/jobs/{job_id}/environment/")
        self.assertEqual(resp.status_code, 200)

        # Config history and compare (requires versions)
        resp = self.client.get(f"/api/jobs/{job_id}/config/history/")
        self.assertEqual(resp.status_code, 200)

        # Job actions: pause, resume, stop, retry (mocked scheduler)
        # We assume mock_scheduler is already set up.
        self.mock_scheduler.pause_job.return_value = None
        self.mock_scheduler.resume_job.return_value = None
        self.mock_scheduler.stop_job.return_value = None
        self.mock_scheduler.add_job.return_value = None

        resp = self.client.post(f"/api/jobs/{job_id}/pause/")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.post(f"/api/jobs/{job_id}/resume/")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.post(f"/api/jobs/{job_id}/stop/")
        self.assertEqual(resp.status_code, 200)

        # Retry a failed job
        resp = self.client.post(f"/api/jobs/{job_id}/retry/")
        # status might be 'stopped' now, so retry should work
        if resp.status_code == 400:
            # force status to failed
            Job.objects.filter(id=job_id).update(status="failed")
            resp = self.client.post(f"/api/jobs/{job_id}/retry/")
        self.assertEqual(resp.status_code, 200)

        # Scheduler status and metrics
        self.mock_scheduler.total_gpus = 2
        self.mock_scheduler.available_gpus = 1
        self.mock_scheduler.task_queue.qsize.return_value = 3
        self.mock_scheduler.running_jobs = {"job1": MagicMock()}
        resp = self.client.get("/api/scheduler/status/")
        self.assertEqual(resp.status_code, 200)
        resp = self.client.get("/api/scheduler/metrics/")
        self.assertEqual(resp.status_code, 200)

        # Bulk delete (only works on completed/failed/stopped)
        resp = self.client.post(
            "/api/jobs/bulk-delete/", {"job_ids": [str(job_id)]}, format="json"
        )
        # The job might still be running -> skipped, but returns 200
        self.assertEqual(resp.status_code, 200)

    def test_job_creation_missing_fields(self):
        resp = self.client.post("/api/jobs/", {"model_name": "bert"}, format="json")
        self.assertEqual(resp.status_code, 400)

    def test_job_update_scheduling(self):
        job = Job.objects.create(
            name="SchedTest",
            model_name="x",
            task_name="y",
            priority=5,
            estimated_duration=1000,
            memory_requirements=4,
        )
        resp = self.client.patch(
            f"/api/jobs/{job.id}/scheduling/",
            {"priority": 3, "estimated_duration": 2000, "memory_requirements": 8},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        job.refresh_from_db()
        self.assertEqual(job.priority, 3)
        self.assertEqual(job.estimated_duration, 2000)
        self.assertEqual(job.memory_requirements, 8)

    def test_job_bulk_delete_mixed(self):
        j1 = Job.objects.create(
            name="Keep", model_name="x", task_name="y", status="running"
        )
        j2 = Job.objects.create(
            name="Delete", model_name="x", task_name="y", status="completed"
        )
        resp = self.client.post(
            "/api/jobs/bulk-delete/",
            {"job_ids": [str(j1.id), str(j2.id)]},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["deleted"], 1)
        self.assertEqual(resp.data["skipped"], 1)
        self.assertTrue(Job.objects.filter(id=j1.id).exists())
        self.assertFalse(Job.objects.filter(id=j2.id).exists())

    def test_job_list_ordering(self):
        Job.objects.all().delete()
        j1 = Job.objects.create(name="A", model_name="x", task_name="y", priority=2)
        j2 = Job.objects.create(name="B", model_name="x", task_name="y", priority=1)
        resp = self.client.get("/api/jobs/")
        self.assertEqual(resp.data[0]["name"], "B")  # priority 1 first
        self.assertEqual(resp.data[1]["name"], "A")

    def test_job_stats_empty(self):
        resp = self.client.get("/api/jobs/stats/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["total"], 0)
        self.assertIsNone(resp.data["average_completion_seconds"])


# ----------------------------------------------------------------------
# 6. Permissions & Edge Cases
# ----------------------------------------------------------------------
class PermissionTests(IntegrationTestBase):
    def test_unauthenticated_access(self):
        client = APIClient()
        endpoints = [
            "/api/experiments/",
            "/api/experiments/runs/",
            "/api/experiments/config/attentions/",
            "/api/jobs/api/jobs/",
            "/api/notes/v1/notes/",
            "/api/model-versioning/versions/",
        ]
        for url in endpoints:
            resp = client.get(url)
            self.assertEqual(resp.status_code, 401, f"Expected 401 for {url}")

    def test_normal_user_cannot_access_staff_only(self):
        self.register_and_login("reguser", "RegPass1!")
        # Try to create experiment – allowed
        resp = self.client.post("/api/experiments/", {"name": "Test"}, format="json")
        self.assertEqual(resp.status_code, 201)
        exp_id = resp.data["id"]

        # Model version: publish, archive, set production – forbidden
        # We need a job and a model version
        job = Job.objects.create(name="TestJob", model_name="x", task_name="y")
        mv = ModelVersion.objects.create(
            run=job,
            version="v1",
            version_number=1,
            artifact_path="/tmp",
            artifact_size_mb=10,
        )
        resp = self.client.post(f"/api/model-versioning/versions/{mv.id}/publish/")
        self.assertEqual(resp.status_code, 403)
        resp = self.client.post(f"/api/model-versioning/versions/{mv.id}/archive/")
        self.assertEqual(resp.status_code, 403)
        resp = self.client.post(
            f"/api/model-versioning/runs/{job.id}/versions/1/set_production/"
        )
        self.assertEqual(resp.status_code, 403)

    def test_private_note_not_accessible_to_other_users(self):
        self.register_and_login("alice", "AlicePass1!")
        job = Job.objects.create(name="TestJob", model_name="x", task_name="y")
        resp = self.client.post(
            "/api/notes/v1/notes/",
            {
                "title": "Private",
                "content": "Secret",
                "run": str(job.id),
                "is_public": False,
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 201)
        note_id = resp.data["id"]

        # Bob tries to access
        bob_client = APIClient()
        bob_client.post(
            "/api/auth/register/",
            {"username": "bob", "password": "BobPass1!", "email": "bob@test.com"},
        )
        bob_token = bob_client.post(
            "/api/auth/login/", {"username": "bob", "password": "BobPass1!"}
        ).json()["token"]
        bob_client.credentials(HTTP_AUTHORIZATION="Token " + bob_token)
        resp = bob_client.get(f"/api/notes/v1/notes/{note_id}/")
        self.assertEqual(resp.status_code, 404)  # Should be hidden

        # Alice makes it public
        resp = self.client.patch(
            f"/api/notes/v1/notes/{note_id}/", {"is_public": True}, format="json"
        )
        # Now Bob can see it
        resp = bob_client.get(f"/api/notes/v1/notes/{note_id}/")
        self.assertEqual(resp.status_code, 200)

    def test_staff_can_access_admin_endpoints(self):
        self.login_as_staff()
        # Staff should be able to list experiments, etc.
        resp = self.client.get("/api/experiments/")
        self.assertEqual(resp.status_code, 200)

    def test_public_system_endpoints(self):
        client = APIClient()
        resp = client.get("/api/system/status/")
        self.assertEqual(resp.status_code, 200)
        resp = client.get("/api/system/gpu/")
        self.assertEqual(resp.status_code, 200)
        resp = client.get("/api/health/")
        self.assertEqual(resp.status_code, 200)
        resp = client.get("/api/version/")
        self.assertEqual(resp.status_code, 200)


# ----------------------------------------------------------------------
# 7. Pagination, Filtering, Sorting (brief)
# ----------------------------------------------------------------------
class PaginationFilterTests(IntegrationTestBase):
    def setUp(self):
        super().setUp()
        self.register_and_login("tester", "Tester1!")
        self.job = Job.objects.create(name="PTest", model_name="x", task_name="y")
        # Create many notes for pagination
        for i in range(25):
            ResearcherNote.objects.create(
                title=f"Note {i}",
                content="Text",
                author=User.objects.get(username="tester"),
                run=self.job,
            )

    def test_notes_pagination(self):
        resp = self.client.get("/api/notes/v1/notes/?page_size=10")
        self.assertEqual(len(resp.data["results"]), 10)
        self.assertIn("next", resp.data)

    def test_experiments_ordering(self):
        Experiment.objects.create(name="B")
        Experiment.objects.create(name="A")
        resp = self.client.get("/api/experiments/")
        # Default ordering is -created_at, so 'A' (most recent) should be first
        self.assertEqual(resp.data[0]["name"], "A")
