import json
import uuid
from unittest.mock import patch, MagicMock

from django.test import TestCase, TransactionTestCase
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone
from rest_framework.test import APIClient
from rest_framework import status
from channels.testing import WebsocketCommunicator
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

from .models import Experiment, TrainingRun, TrainingSnapshot
from .serializers import ExperimentSerializer, TrainingRunSerializer, TrainingSnapshotSerializer
from .jobs import start_job, stop_job, stop_all_jobs, push_update
from .runners import run_training, run_evaluation, run_compression
from django_api.asgi import application  # adjust if necessary
from unittest.mock import patch

# ----------------------------------------------------------------------
def create_user_and_client():
    user = User.objects.create_user(username='testuser', password='testpass')
    client = APIClient()
    client.force_authenticate(user=user)
    return user, client

# ----------------------------------------------------------------------
class ExperimentModelTests(TestCase):
    def test_create_experiment(self):
        exp = Experiment.objects.create(name="Test Experiment", tags="nlp,test")
        self.assertIsInstance(exp.id, uuid.UUID)
        self.assertEqual(str(exp), "Test Experiment")
        self.assertEqual(exp.tags, "nlp,test")
        self.assertIsNotNone(exp.created_at)

    def test_ordering(self):
        Experiment.objects.create(name="Older")
        Experiment.objects.create(name="Newer")
        self.assertEqual(list(Experiment.objects.all().order_by('-created_at'))[0].name, "Newer")

class TrainingRunModelTests(TestCase):
    def setUp(self):
        self.exp = Experiment.objects.create(name="Exp1")

    def test_create_run(self):
        run = TrainingRun.objects.create(experiment=self.exp, name="Run1")
        self.assertEqual(run.status, "queued")
        self.assertTrue(run.id)

    def test_str(self):
        run = TrainingRun.objects.create(experiment=self.exp, name="My Run", job_type="evaluation")
        self.assertIn(self.exp.name, str(run))

class TrainingSnapshotModelTests(TestCase):
    def setUp(self):
        self.exp = Experiment.objects.create(name="Exp1")
        self.run = TrainingRun.objects.create(experiment=self.exp)

    def test_create_snapshot(self):
        snap = TrainingSnapshot.objects.create(run=self.run, step=10, train_loss=0.5)
        self.assertEqual(snap.step, 10)

    def test_unique_together(self):
        TrainingSnapshot.objects.create(run=self.run, step=1)
        with self.assertRaises(Exception):
            TrainingSnapshot.objects.create(run=self.run, step=1)

# ----------------------------------------------------------------------
class SerializerTests(TestCase):
    def setUp(self):
        self.exp = Experiment.objects.create(name="Serializer Exp")
        self.run = TrainingRun.objects.create(experiment=self.exp, name="Run")

    def test_experiment_serializer(self):
        data = ExperimentSerializer(self.exp).data
        self.assertEqual(data['name'], "Serializer Exp")
        self.assertIn('runs_count', data)

    def test_training_run_serializer(self):
        data = TrainingRunSerializer(self.run).data
        self.assertEqual(data['experiment_name'], "Serializer Exp")

    def test_training_snapshot_serializer(self):
        snap = TrainingSnapshot.objects.create(run=self.run, step=5, train_loss=0.2)
        data = TrainingSnapshotSerializer(snap).data
        self.assertEqual(data['step'], 5)

# ----------------------------------------------------------------------
class ExperimentViewTests(TestCase):
    def setUp(self):
        self.user, self.client = create_user_and_client()

    def test_list_experiments(self):
        Experiment.objects.create(name="E1")
        Experiment.objects.create(name="E2")
        response = self.client.get('/api/experiments/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 2)

    def test_create_experiment(self):
        response = self.client.post('/api/experiments/create/', {'name': 'NewExp', 'tags': 'demo'}, format='json')
        self.assertEqual(response.status_code, 201)
        self.assertEqual(Experiment.objects.count(), 1)

    def test_experiment_detail(self):
        exp = Experiment.objects.create(name="Detail Exp")
        response = self.client.get(f'/api/experiments/{exp.id}/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['name'], "Detail Exp")

# Add this import at the top if not already present

class TrainingRunViewTests(TestCase):
    def setUp(self):
        self.user, self.client = create_user_and_client()
        self.exp = Experiment.objects.create(name="RunTestExp")
        self.run = TrainingRun.objects.create(experiment=self.exp, name="TestRun")

        # Patch start_job to run the target function instantly (no threads)
        patcher = patch('experiment.views.start_job')
        self.mock_start_job = patcher.start()
        def run_sync(run, target_func):
            # Simulate synchronous execution
            target_func(run, stop_event=None)
        self.mock_start_job.side_effect = run_sync
        self.addCleanup(patcher.stop)

    def test_list_runs(self):
        response = self.client.get('/api/runs/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)

    def test_run_detail(self):
        response = self.client.get(f'/api/runs/{self.run.id}/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['status'], 'queued')

    def test_start_train(self):
        response = self.client.post('/api/train/', {
            'experiment_id': str(self.exp.id),
            'name': 'Train1',
            'job_type': 'training',
            'config': {'epochs': 1}
        }, format='json')
        self.assertEqual(response.status_code, 202)
        self.assertEqual(TrainingRun.objects.count(), 2)

    def test_start_evaluate_with_url(self):
        response = self.client.post(f'/api/runs/{self.run.id}/evaluate/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('status', response.data)

    def test_stop_job(self):
        self.run.status = 'running'
        self.run.save()
        response = self.client.post(f'/api/runs/{self.run.id}/stop/')
        self.assertEqual(response.status_code, 200)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, 'stopped')

    def test_training_history(self):
        self.run.training_history = {"loss": [0.5, 0.3]}
        self.run.save()
        response = self.client.get(f'/api/runs/{self.run.id}/history/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['loss'], [0.5, 0.3])

    def test_snapshots(self):
        TrainingSnapshot.objects.create(run=self.run, step=1, train_loss=0.9)
        response = self.client.get(f'/api/runs/{self.run.id}/snapshots/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)

    def test_update_config(self):
        response = self.client.patch(f'/api/runs/{self.run.id}/config/', {'learning_rate': 0.001}, format='json')
        self.assertEqual(response.status_code, 200)
        self.run.refresh_from_db()
        self.assertEqual(self.run.config['learning_rate'], 0.001)

    def test_stop_all_jobs(self):
        response = self.client.post('/api/jobs/stop_all/')
        self.assertEqual(response.status_code, 200)


class SystemViewTests(TestCase):
    def setUp(self):
        self.user, self.client = create_user_and_client()

    def test_system_status(self):
        response = self.client.get('/api/system/status/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('cpu', response.data)

    def test_gpu_info(self):
        response = self.client.get('/api/system/gpu/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('type', response.data)

class AuthViewTests(TestCase):
    def test_register(self):
        client = APIClient()
        response = client.post('/api/auth/register/', {'username': 'newuser', 'password': 'newpass123'})
        self.assertEqual(response.status_code, 201)
        self.assertIn('token', response.data)

    def test_login(self):
        User.objects.create_user(username='loginuser', password='loginpass')
        client = APIClient()
        response = client.post('/api/auth/login/', {'username': 'loginuser', 'password': 'loginpass'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('token', response.data)

    def test_logout(self):
        user = User.objects.create_user(username='logoutuser', password='logoutpass')
        client = APIClient()
        client.force_authenticate(user=user)   # must be authenticated to log out
        response = client.post('/api/auth/logout/')
        self.assertEqual(response.status_code, 200)
        from rest_framework.authtoken.models import Token
        self.assertFalse(Token.objects.filter(user=user).exists())

# ----------------------------------------------------------------------
class RunnerTests(TestCase):
    def setUp(self):
        self.exp = Experiment.objects.create(name="RunnerExp")
        self.run = TrainingRun.objects.create(experiment=self.exp, job_type="training")
        self.stop_event = MagicMock()
        self.stop_event.is_set.return_value = False

    @patch('experiment.runners.push_update')
    @patch('experiment.runners._log')
    def test_run_training_completes(self, mock_log, mock_push):
        run_training(self.run, self.stop_event)
        self.run.refresh_from_db()
        self.assertIn(self.run.status, ['completed', 'running'])

    @patch('experiment.runners.push_update')
    @patch('experiment.runners._log')
    def test_run_training_stopped(self, mock_log, mock_push):
        self.stop_event.is_set.return_value = True
        run_training(self.run, self.stop_event)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, 'stopped')

    @patch('experiment.runners.push_update')
    @patch('experiment.runners._log')
    def test_run_evaluation(self, mock_log, mock_push):
        run_evaluation(self.run, self.stop_event)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, 'completed')

    @patch('experiment.runners.push_update')
    @patch('experiment.runners._log')
    def test_run_compression(self, mock_log, mock_push):
        run_compression(self.run, self.stop_event)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, 'completed')

# ----------------------------------------------------------------------
class JobConsumerTests(TransactionTestCase):
    def setUp(self):
        self.exp = Experiment.objects.create(name="WSEXP")
        self.run = TrainingRun.objects.create(experiment=self.exp, name="WSRun")

    def test_connect_to_job(self):
        async def test():
            communicator = WebsocketCommunicator(application, f"/ws/jobs/{self.run.id}/")
            connected, _ = await communicator.connect()
            self.assertTrue(connected)
            await communicator.disconnect()
        async_to_sync(test)()

    def test_receive_job_update(self):
        async def test():
            communicator = WebsocketCommunicator(application, f"/ws/jobs/{self.run.id}/")
            connected, _ = await communicator.connect()
            self.assertTrue(connected)
            channel_layer = get_channel_layer()
            await channel_layer.group_send(
                f"job_{self.run.id}",
                {"type": "job_update", "data": {"progress": 50}}
            )
            response = await communicator.receive_json_from()
            self.assertEqual(response, {"progress": 50})
            await communicator.disconnect()
        async_to_sync(test)()

    def test_terminal_consumer(self):
        async def test():
            communicator = WebsocketCommunicator(application, "/ws/terminal/")
            connected, _ = await communicator.connect()
            self.assertTrue(connected)
            channel_layer = get_channel_layer()
            await channel_layer.group_send(
                "terminal",
                {"type": "terminal_message", "data": "Hello from test"}
            )
            response = await communicator.receive_from()
            # Accept either a plain string or JSON with a 'data' key
            if response.startswith('{'):
                import json
                event = json.loads(response)
                self.assertEqual(event['data'], "Hello from test")
            else:
                self.assertEqual(response, "Hello from test")
            await communicator.disconnect()
        async_to_sync(test)()





# ----------------------------------------------------------------------
# ADDITIONAL HARD TESTS (30) – insert into existing test classes
# ----------------------------------------------------------------------

# ======== ExperimentModelTests additions ========
def test_experiment_defaults(self):
    exp = Experiment.objects.create(name="Default Exp")
    self.assertEqual(exp.description, "")
    self.assertEqual(exp.default_config, {})
    self.assertEqual(exp.tags, "")

def test_experiment_name_max_length(self):
    long_name = "a" * 201   # max_length is 200
    with self.assertRaises(Exception):
        Experiment.objects.create(name=long_name)


# ======== TrainingRunModelTests additions ========
def test_run_status_default(self):
    run = TrainingRun.objects.create(experiment=self.exp)
    self.assertEqual(run.status, "queued")

def test_run_invalid_status_choice(self):
    with self.assertRaises(Exception):
        TrainingRun.objects.create(experiment=self.exp, status="invalid_status")


# ======== TrainingSnapshotModelTests additions ========
def test_snapshot_optional_fields(self):
    snap = TrainingSnapshot.objects.create(run=self.run, step=1,
                                           cpu_percent=None, ram_mb=None)
    self.assertIsNone(snap.cpu_percent)
    self.assertIsNone(snap.ram_mb)


# ======== ExperimentViewTests additions ========
def test_create_experiment_missing_name(self):
    response = self.client.post('/api/experiments/create/', {}, format='json')
    self.assertEqual(response.status_code, 400)

def test_experiment_detail_not_found(self):
    fake_id = '00000000-0000-0000-0000-000000000000'
    response = self.client.get(f'/api/experiments/{fake_id}/')
    self.assertEqual(response.status_code, 404)

def test_list_experiments_unauthenticated(self):
    client = APIClient()   # no authentication
    response = client.get('/api/experiments/')
    self.assertEqual(response.status_code, 401)   # IsAuthenticated required


# ======== TrainingRunViewTests additions ========
# (these methods already have start_job patched in setUp)
def test_start_train_missing_experiment_id(self):
    response = self.client.post('/api/train/', {
        'name': 'NoExperiment',
        'config': {'epochs': 1}
    }, format='json')
    # The view creates a new experiment if none provided? Actually it only gets experiment_id.
    # If experiment_id is missing, the view will try to get_object_or_404 with None -> 404.
    self.assertEqual(response.status_code, 404)

def test_start_evaluate_missing_run_id_in_body(self):
    # The route /evaluate/ expects run_id in request.data
    response = self.client.post('/api/evaluate/', {}, format='json')
    # Without run_id, get_object_or_404 raises 404
    self.assertEqual(response.status_code, 404)

def test_start_compress_missing_run_id(self):
    response = self.client.post('/api/compress/', {}, format='json')
    self.assertEqual(response.status_code, 404)

def test_start_evaluate_invalid_run_id(self):
    response = self.client.post('/api/runs/invalid-uuid/evaluate/')
    self.assertEqual(response.status_code, 404)

def test_stop_job_already_completed(self):
    self.run.status = 'completed'
    self.run.save()
    response = self.client.post(f'/api/runs/{self.run.id}/stop/')
    # View returns 400 because status is not queued/running
    self.assertEqual(response.status_code, 400)

def test_update_config_with_invalid_json(self):
    response = self.client.patch(f'/api/runs/{self.run.id}/config/',
                                 'not json', content_type='application/json')
    self.assertEqual(response.status_code, 400)

def test_training_history_nonexistent_run(self):
    response = self.client.get('/api/runs/00000000-0000-0000-0000-000000000000/history/')
    self.assertEqual(response.status_code, 404)

def test_snapshots_nonexistent_run(self):
    response = self.client.get('/api/runs/00000000-0000-0000-0000-000000000000/snapshots/')
    self.assertEqual(response.status_code, 404)

def test_run_artifacts_empty(self):
    # run with no artifacts_dir set
    response = self.client.get(f'/api/runs/{self.run.id}/artifacts/')
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.data['artifacts'], [])

def test_run_artifact_download_nonexistent(self):
    # run has no artifacts_dir
    response = self.client.get(f'/api/runs/{self.run.id}/artifacts/somefile')
    self.assertEqual(response.status_code, 404)

def test_run_model_download_no_model(self):
    # run with no artifacts_dir/model
    response = self.client.get(f'/api/runs/{self.run.id}/model/')
    self.assertEqual(response.status_code, 404)

def test_list_runs_unauthenticated(self):
    client = APIClient()
    response = client.get('/api/runs/')
    self.assertEqual(response.status_code, 401)

def test_generate_paper_not_implemented(self):
    response = self.client.get(f'/api/runs/{self.run.id}/paper/')
    # The placeholder returns 200 with message
    self.assertEqual(response.status_code, 200)
    self.assertIn('message', response.data)

def test_paper_preview_pdf_not_implemented(self):
    response = self.client.get(f'/api/runs/{self.run.id}/paper_preview/')
    self.assertEqual(response.status_code, 501)


# ======== SystemViewTests additions ========
def test_system_status_unauthenticated(self):
    client = APIClient()
    response = client.get('/api/system/status/')
    # system_status view has no permission_classes, so public
    self.assertEqual(response.status_code, 200)

def test_gpu_info_unauthenticated(self):
    client = APIClient()
    response = client.get('/api/system/gpu/')
    self.assertEqual(response.status_code, 200)


# ======== AuthViewTests additions ========
def test_register_missing_fields(self):
    client = APIClient()
    response = client.post('/api/auth/register/', {'username': 'onlyuser'})
    self.assertEqual(response.status_code, 400)

def test_login_missing_fields(self):
    client = APIClient()
    response = client.post('/api/auth/login/', {'username': 'user'})
    self.assertEqual(response.status_code, 401)

def test_logout_already_logged_out(self):
    user = User.objects.create_user(username='nouser', password='nopass')
    client = APIClient()
    client.force_authenticate(user=user)   # user has no token
    response = client.post('/api/auth/logout/')
    # The view should not crash; it filters tokens and deletes none
    self.assertEqual(response.status_code, 200)


# ======== RunnerTests additions ========
@patch('experiment.runners.push_update')
@patch('experiment.runners._log')
def test_run_training_exception(self, mock_log, mock_push):
    # Force the training stub to raise
    with patch('experiment.runners.framework_train', side_effect=Exception("Test error")):
        run_training(self.run, self.stop_event)
    self.run.refresh_from_db()
    self.assertEqual(self.run.status, 'failed')

@patch('experiment.runners.push_update')
@patch('experiment.runners._log')
def test_run_evaluation_exception(self, mock_log, mock_push):
    with patch('experiment.runners.framework_eval', side_effect=Exception("Eval error")):
        run_evaluation(self.run, self.stop_event)
    self.run.refresh_from_db()
    self.assertEqual(self.run.status, 'failed')

@patch('experiment.runners.push_update')
@patch('experiment.runners._log')
def test_run_compression_exception(self, mock_log, mock_push):
    with patch('experiment.runners.framework_compress', side_effect=Exception("Compress error")):
        run_compression(self.run, self.stop_event)
    self.run.refresh_from_db()
    self.assertEqual(self.run.status, 'failed')


# ======== ConsumerTests additions ========
def test_connect_to_invalid_run(self):
    async def test():
        communicator = WebsocketCommunicator(application, "/ws/jobs/00000000-0000-0000-0000-000000000000/")
        connected, _ = await communicator.connect()
        # Should still connect because the consumer just joins a group (no validation)
        self.assertTrue(connected)
        await communicator.disconnect()
    async_to_sync(test)()

def test_receive_multiple_job_updates(self):
    async def test():
        communicator = WebsocketCommunicator(application, f"/ws/jobs/{self.run.id}/")
        await communicator.connect()
        channel_layer = get_channel_layer()
        await channel_layer.group_send(
            f"job_{self.run.id}",
            {"type": "job_update", "data": {"step": 1}}
        )
        await channel_layer.group_send(
            f"job_{self.run.id}",
            {"type": "job_update", "data": {"step": 2}}
        )
        msg1 = await communicator.receive_json_from()
        msg2 = await communicator.receive_json_from()
        self.assertEqual(msg1, {"step": 1})
        self.assertEqual(msg2, {"step": 2})
        await communicator.disconnect()
    async_to_sync(test)()

def test_terminal_consumer_disconnect(self):
    async def test():
        communicator = WebsocketCommunicator(application, "/ws/terminal/")
        connected, _ = await communicator.connect()
        self.assertTrue(connected)
        await communicator.disconnect()
        # After disconnect, sending to group should not reach
        channel_layer = get_channel_layer()
        await channel_layer.group_send(
            "terminal",
            {"type": "terminal_message", "data": "after disconnect"}
        )
        # communicator.receive_from would timeout; we accept a short timeout
        with self.assertRaises(Exception):
            await communicator.receive_from()
    async_to_sync(test)()