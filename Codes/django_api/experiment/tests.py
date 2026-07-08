import json
import uuid
from unittest.mock import MagicMock, patch

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from channels.testing import WebsocketCommunicator
from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import TestCase, TransactionTestCase, override_settings
from django_api.asgi import application
from experiment.models import Experiment, TrainingRun, TrainingSnapshot
from experiment.runners import run_compression, run_evaluation, run_training
from experiment.serializers import (ExperimentSerializer,
                                    TrainingRunSerializer,
                                    TrainingSnapshotSerializer)
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------
def create_authenticated_client(user=None):
    if user is None:
        user = User.objects.create_user(username="testuser", password="testpass")
    token, _ = Token.objects.get_or_create(user=user)
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION=f"Token {token.key}")
    client.cookies["authToken"] = token.key
    return client, user, token


def clear_throttle_cache():
    cache.clear()


# ============================= MODEL TESTS =============================
class ExperimentModelTests(TestCase):
    def test_create_experiment(self):
        exp = Experiment.objects.create(name="Test", tags="nlp")
        self.assertIsInstance(exp.id, uuid.UUID)
        self.assertEqual(str(exp), "Test")

    def test_ordering(self):
        Experiment.objects.create(name="Old")
        Experiment.objects.create(name="New")
        names = list(
            Experiment.objects.all()
            .order_by("-created_at")
            .values_list("name", flat=True)
        )
        self.assertEqual(names, ["New", "Old"])


class TrainingRunModelTests(TestCase):
    def setUp(self):
        self.exp = Experiment.objects.create(name="Exp")

    def test_create_run(self):
        run = TrainingRun.objects.create(experiment=self.exp, name="Run")
        self.assertEqual(run.status, "queued")
        self.assertTrue(run.id)


class TrainingSnapshotModelTests(TestCase):
    def setUp(self):
        self.exp = Experiment.objects.create(name="Exp")
        self.run = TrainingRun.objects.create(experiment=self.exp)

    def test_create_snapshot(self):
        snap = TrainingSnapshot.objects.create(run=self.run, step=10, train_loss=0.5)
        self.assertEqual(snap.step, 10)

    def test_unique_together(self):
        TrainingSnapshot.objects.create(run=self.run, step=1)
        with self.assertRaises(Exception):
            TrainingSnapshot.objects.create(run=self.run, step=1)


# ========================== SERIALIZER TESTS ==========================
class SerializerTests(TestCase):
    def setUp(self):
        self.exp = Experiment.objects.create(name="Exp")
        self.run = TrainingRun.objects.create(experiment=self.exp, name="Run")

    def test_experiment_serializer(self):
        data = ExperimentSerializer(self.exp).data
        self.assertIn("runs_count", data)

    def test_run_serializer(self):
        data = TrainingRunSerializer(self.run).data
        self.assertEqual(data["experiment_name"], "Exp")

    def test_snapshot_serializer(self):
        snap = TrainingSnapshot.objects.create(run=self.run, step=5, train_loss=0.2)
        data = TrainingSnapshotSerializer(snap).data
        self.assertEqual(data["step"], 5)


# ========================== AUTH ENDPOINTS ============================
class AuthEndpointTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.register_url = "/api/auth/register/"
        self.login_url = "/api/auth/login/"
        self.logout_url = "/api/auth/logout/"
        self.profile_url = "/api/auth/profile/"
        self.password_url = "/api/auth/change-password/"
        self.refresh_url = "/api/auth/refresh-token/"
        self.user_info_url = "/api/auth/user/"
        self.test_user = User.objects.create_user(
            username="exist", password="existpass", email="exist@test.com"
        )
        self.token = Token.objects.create(user=self.test_user)
        clear_throttle_cache()

    def _authenticate(self):
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {self.token.key}")
        self.client.cookies["authToken"] = self.token.key

    def test_register_success(self):
        resp = self.client.post(
            self.register_url,
            {"username": "newuser", "password": "Strong1", "email": "new@e.com"},
        )
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        self.assertIn("token", resp.data)
        self.assertIn("authToken", resp.cookies)

    def test_register_missing_fields(self):
        resp = self.client.post(self.register_url, {"username": "x"})
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_register_short_username(self):
        resp = self.client.post(
            self.register_url, {"username": "ab", "password": "Strong1"}
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_register_short_password(self):
        resp = self.client.post(
            self.register_url, {"username": "user", "password": "12345"}
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_register_duplicate_username(self):
        resp = self.client.post(
            self.register_url, {"username": "exist", "password": "Strong1"}
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_register_invalid_email(self):
        resp = self.client.post(
            self.register_url,
            {"username": "u", "password": "Strong1", "email": "bademail"},
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_register_long_username(self):
        resp = self.client.post(
            self.register_url, {"username": "a" * 151, "password": "Strong1"}
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_login_success(self):
        resp = self.client.post(
            self.login_url, {"username": "exist", "password": "existpass"}
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data["token"], self.token.key)
        self.assertIn("authToken", resp.cookies)

    def test_login_invalid_password(self):
        resp = self.client.post(
            self.login_url, {"username": "exist", "password": "wrong"}
        )
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_login_nonexistent_user(self):
        resp = self.client.post(self.login_url, {"username": "nobody", "password": "x"})
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_login_missing_fields(self):
        resp = self.client.post(self.login_url, {"username": "exist"})
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_login_empty_payload(self):
        resp = self.client.post(self.login_url, {})
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_logout_success(self):
        self._authenticate()
        resp = self.client.post(self.logout_url)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertFalse(Token.objects.filter(key=self.token.key).exists())

    def test_logout_unauthenticated(self):
        resp = self.client.post(self.logout_url)
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_refresh_token(self):
        self._authenticate()
        resp = self.client.post(self.refresh_url)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertNotEqual(resp.data["token"], self.token.key)

    def test_change_password_success(self):
        self._authenticate()
        resp = self.client.post(
            self.password_url, {"old_password": "existpass", "new_password": "NewPass1"}
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.test_user.refresh_from_db()
        self.assertTrue(self.test_user.check_password("NewPass1"))

    def test_change_password_wrong_old(self):
        self._authenticate()
        resp = self.client.post(
            self.password_url, {"old_password": "wrong", "new_password": "NewPass1"}
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_update_profile_success(self):
        self._authenticate()
        resp = self.client.patch(self.profile_url, {"first_name": "John"})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.test_user.refresh_from_db()
        self.assertEqual(self.test_user.first_name, "John")

    def test_update_profile_invalid_email(self):
        self._authenticate()
        resp = self.client.patch(self.profile_url, {"email": "bademail"})
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_get_user_info(self):
        self._authenticate()
        resp = self.client.get(self.user_info_url)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data["username"], "exist")


# ========================== CONFIG ENDPOINTS ==========================
class ConfigEndpointTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="conf", password="pass")
        self.client, _, _ = create_authenticated_client(self.user)

    def test_list_attentions(self):
        resp = self.client.get("/api/config/attentions/")
        self.assertEqual(resp.status_code, 200)

    def test_list_compressors(self):
        resp = self.client.get("/api/config/compressors/")
        self.assertEqual(resp.status_code, 200)

    def test_list_models(self):
        resp = self.client.get("/api/config/models/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("distilbert-base-uncased", resp.data["models"])

    def test_list_tasks(self):
        resp = self.client.get("/api/config/tasks/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("mrpc", resp.data["tasks"])

    def test_list_methods(self):
        resp = self.client.get("/api/config/methods/")
        self.assertEqual(resp.status_code, 200)

    def test_list_presets(self):
        resp = self.client.get("/api/config/presets/")
        self.assertEqual(resp.status_code, 200)

    def test_config_requires_auth(self):
        client = APIClient()
        resp = client.get("/api/config/attentions/")
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)


# ========================== SYSTEM ENDPOINTS ==========================
class SystemEndpointTests(TestCase):
    def test_system_status_public(self):
        resp = APIClient().get("/api/system/status/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("cpu", resp.data)

    def test_gpu_info_public(self):
        resp = APIClient().get("/api/system/gpu/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("type", resp.data)

    def test_health_check(self):
        resp = APIClient().get("/api/health/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["status"], "healthy")

    def test_version(self):
        resp = APIClient().get("/api/version/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("api_version", resp.data)


# ========================== EXPERIMENT VIEWS ==========================
class ExperimentViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="expuser", password="pass")
        self.client, _, _ = create_authenticated_client(self.user)

    def test_create_experiment(self):
        resp = self.client.post(
            "/api/experiments/", {"name": "Exp1", "tags": "test"}, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Experiment.objects.count(), 1)

    def test_list_experiments(self):
        Experiment.objects.create(name="E1")
        Experiment.objects.create(name="E2")
        resp = self.client.get("/api/experiments/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.data), 2)

    def test_experiment_detail(self):
        exp = Experiment.objects.create(name="Detail")
        resp = self.client.get(f"/api/experiments/{exp.id}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["name"], "Detail")

    def test_experiment_detail_not_found(self):
        resp = self.client.get("/api/experiments/00000000-0000-0000-0000-000000000000/")
        self.assertEqual(resp.status_code, 404)

    def test_delete_experiment(self):
        exp = Experiment.objects.create(name="Del")
        resp = self.client.delete(f"/api/experiments/{exp.id}/")
        self.assertEqual(resp.status_code, 204)
        self.assertFalse(Experiment.objects.filter(id=exp.id).exists())


# ========================== TRAINING RUN VIEWS ========================
class TrainingRunViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="runuser", password="pass")
        self.client, _, _ = create_authenticated_client(self.user)
        self.exp = Experiment.objects.create(name="RunExp")
        # Patch the real start_job to run synchronously
        patcher = patch("experiment.jobs.start_job")
        self.mock_start_job = patcher.start()

        def sync_start(run, target):
            target(run, None)  # run runner immediately

        self.mock_start_job.side_effect = sync_start
        self.addCleanup(patcher.stop)

    def test_start_train_new_experiment(self):
        resp = self.client.post(
            "/api/train/", {"name": "Training", "config": {"epochs": 1}}, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        run = TrainingRun.objects.first()
        self.assertEqual(run.job_type, "training")
        self.assertIn(run.status, ["completed", "failed", "running"])

    def test_start_train_existing_experiment(self):
        resp = self.client.post(
            "/api/train/",
            {"experiment_id": str(self.exp.id), "config": {"epochs": 1}},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        self.assertTrue(TrainingRun.objects.filter(experiment=self.exp).exists())

    def test_start_evaluate(self):
        resp = self.client.post(
            "/api/evaluate/",
            {"experiment_id": str(self.exp.id), "config": {}},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        run = TrainingRun.objects.filter(job_type="evaluation").first()
        self.assertIsNotNone(run)

    def test_start_compress(self):
        resp = self.client.post(
            "/api/compress/",
            {"experiment_id": str(self.exp.id), "config": {}},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)

    def test_run_detail(self):
        run = TrainingRun.objects.create(experiment=self.exp, name="Run1")
        resp = self.client.get(f"/api/runs/{run.id}/")
        self.assertEqual(resp.status_code, 200)

    def test_stop_run(self):
        run = TrainingRun.objects.create(experiment=self.exp, status="running")
        resp = self.client.post(f"/api/runs/{run.id}/stop/")
        self.assertEqual(resp.status_code, 200)
        run.refresh_from_db()
        self.assertEqual(run.status, "stopped")

    def test_stop_run_not_running(self):
        run = TrainingRun.objects.create(experiment=self.exp, status="completed")
        resp = self.client.post(f"/api/runs/{run.id}/stop/")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)


# ========================== COOKIE AUTH TEST ==========================
class CookieAuthTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="cookie", password="pass")
        self.token = Token.objects.create(user=self.user)

    def test_cookie_auth_success(self):
        client = APIClient()
        client.cookies["authToken"] = self.token.key
        resp = client.get("/api/auth/user/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["username"], "cookie")

    def test_cookie_auth_invalid_token(self):
        client = APIClient()
        client.cookies["authToken"] = "invalid"
        resp = client.get("/api/auth/user/")
        self.assertEqual(resp.status_code, 401)

    def test_bearer_header_works(self):
        client = APIClient()
        client.credentials(HTTP_AUTHORIZATION=f"Bearer {self.token.key}")
        resp = client.get("/api/auth/user/")
        self.assertEqual(resp.status_code, 200)


# ========================== RATE LIMIT TESTS ==========================
class RateLimitTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.throttle_patcher = override_settings(
            REST_FRAMEWORK={
                "DEFAULT_THROTTLE_CLASSES": [
                    "rest_framework.throttling.AnonRateThrottle"
                ],
                "DEFAULT_THROTTLE_RATES": {"anon": "2/min"},
                "DEFAULT_AUTHENTICATION_CLASSES": [],
                "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
            }
        )
        self.throttle_patcher.enable()
        self.addCleanup(self.throttle_patcher.disable)
        cache.clear()

    def test_anon_throttle_register(self):
        for i in range(3):
            resp = self.client.post(
                "/api/auth/register/", {"username": f"user{i}", "password": "Strong1"}
            )
        self.assertEqual(resp.status_code, status.HTTP_429_TOO_MANY_REQUESTS)


# ========================== CONSUMER TESTS ============================
class ConsumerTests(TransactionTestCase):
    def setUp(self):
        self.exp = Experiment.objects.create(name="WSExp")
        self.run = TrainingRun.objects.create(experiment=self.exp, name="WSRun")

    def test_connect_to_job_consumer(self):
        async def test():
            communicator = WebsocketCommunicator(
                application, f"/ws/experiments/{self.run.id}/"
            )
            connected, _ = await communicator.connect()
            self.assertTrue(connected)
            await communicator.disconnect()

        async_to_sync(test)()

    def test_receive_job_update(self):
        async def test():
            communicator = WebsocketCommunicator(
                application, f"/ws/experiments/{self.run.id}/"
            )
            await communicator.connect()
            channel_layer = get_channel_layer()
            await channel_layer.group_send(
                f"job_{self.run.id}", {"type": "job_update", "data": {"progress": 50}}
            )
            response = await communicator.receive_json_from()
            self.assertEqual(response, {"progress": 50})
            await communicator.disconnect()

        async_to_sync(test)()

    def test_terminal_consumer(self):
        async def test():
            communicator = WebsocketCommunicator(application, "/ws/terminal/")
            await communicator.connect()
            channel_layer = get_channel_layer()
            await channel_layer.group_send(
                "terminal", {"type": "terminal_message", "data": "Hello"}
            )
            response = await communicator.receive_from()
            data = json.loads(response)
            self.assertEqual(data["data"], "Hello")
            await communicator.disconnect()

        async_to_sync(test)()


# ========================== RUNNER TESTS ==============================
class RunnerTests(TestCase):
    def setUp(self):
        self.exp = Experiment.objects.create(name="RunnerExp")
        self.run = TrainingRun.objects.create(experiment=self.exp, job_type="training")
        self.stop_event = MagicMock()
        self.stop_event.is_set.return_value = False
        # Mock channel functions to avoid DB locks
        patcher1 = patch("experiment.runners.push_update")
        patcher2 = patch("experiment.runners._log")
        self.mock_push = patcher1.start()
        self.mock_log = patcher2.start()
        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)
        # Import the real framework modules and mock them
        self.fw_patcher = patch.multiple(
            "experiment.runners",
            framework_train=MagicMock(),
            framework_eval=MagicMock(return_value={"eval_accuracy": 0.9}),
            framework_compress=MagicMock(return_value={"compression_ratio": 0.5}),
        )
        self.fw_mocks = self.fw_patcher.start()
        self.addCleanup(self.fw_patcher.stop)

    def test_run_training_completes(self):
        run_training(self.run, self.stop_event)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, "completed")

    def test_run_training_stopped(self):
        self.stop_event.is_set.return_value = True
        run_training(self.run, self.stop_event)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, "stopped")

    def test_run_training_fails(self):
        self.fw_mocks["framework_train"].side_effect = Exception("crash")
        run_training(self.run, self.stop_event)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, "failed")

    def test_run_evaluation_completes(self):
        run_evaluation(self.run, self.stop_event)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, "completed")

    def test_run_evaluation_fails(self):
        self.fw_mocks["framework_eval"].side_effect = Exception("eval crash")
        run_evaluation(self.run, self.stop_event)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, "failed")

    def test_run_compression_completes(self):
        run_compression(self.run, self.stop_event)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, "completed")
