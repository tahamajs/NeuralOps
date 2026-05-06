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

class TrainingRunViewTests(TestCase):
    def setUp(self):
        self.user, self.client = create_user_and_client()
        self.exp = Experiment.objects.create(name="RunTestExp")
        self.run = TrainingRun.objects.create(experiment=self.exp, name="TestRun")

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

    @patch('experiment.views.start_job')
    def test_start_train(self, mock_start_job):
        # Make start_job run the training function immediately instead of spawning a thread
        def fake_start_job(run, target_func):
            target_func(run, stop_event=None)
        mock_start_job.side_effect = fake_start_job

        response = self.client.post('/api/train/', {
            'experiment_id': str(self.exp.id),
            'name': 'Train1',
            'job_type': 'training',
            'config': {'epochs': 1}
        }, format='json')
        self.assertEqual(response.status_code, 202)
        self.assertEqual(TrainingRun.objects.count(), 2)



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