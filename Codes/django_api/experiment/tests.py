import json
import uuid
from unittest.mock import patch, MagicMock

from django.test import TestCase, TransactionTestCase, override_settings
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone
from rest_framework.test import APIClient, APITestCase
from rest_framework import status
from rest_framework.authtoken.models import Token
from channels.testing import WebsocketCommunicator
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

from .models import Experiment, TrainingRun, TrainingSnapshot
from .serializers import ExperimentSerializer, TrainingRunSerializer, TrainingSnapshotSerializer
from .jobs import start_job, stop_job, stop_all_jobs, push_update
from .runners import run_training, run_evaluation, run_compression
from django_api.asgi import application  # adjust import if needed

# ----------------------------------------------------------------------
# Helper classes / mixins
# ----------------------------------------------------------------------
class AuthMixin:
    """Mixin to create a user and authenticate the client."""
    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.token = Token.objects.create(user=self.user)
        self.client = APIClient()
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.token.key)


# ----------------------------------------------------------------------
# Model tests
# ----------------------------------------------------------------------
class ExperimentModelTests(TestCase):
    def test_create_experiment(self):
        exp = Experiment.objects.create(name="Test Experiment", tags="nlp,test")
        self.assertIsInstance(exp.id, uuid.UUID)
        self.assertEqual(str(exp), "Test Experiment")
        self.assertEqual(exp.tags, "nlp,test")
        self.assertIsNotNone(exp.created_at)
        self.assertIsNotNone(exp.updated_at)

    def test_ordering(self):
        exp1 = Experiment.objects.create(name="Older")
        exp2 = Experiment.objects.create(name="Newer")
        self.assertEqual(list(Experiment.objects.all()), [exp2, exp1])  # descending by created_at


class TrainingRunModelTests(TestCase):
    def setUp(self):
        self.experiment = Experiment.objects.create(name="Exp1")

    def test_create_run(self):
        run = TrainingRun.objects.create(
            experiment=self.experiment,
            name="Run1",
            job_type="training",
            status="queued"
        )
        self.assertEqual(run.status, "queued")
        self.assertIsNotNone(run.id)
        self.assertEqual(run.experiment, self.experiment)
        self.assertIsNone(run.started_at)
        self.assertIsNone(run.completed_at)

    def test_str(self):
        run = TrainingRun.objects.create(experiment=self.experiment, name="My Run", job_type="evaluation")
        self.assertEqual(str(run), f"{self.experiment.name} - My Run")


class TrainingSnapshotModelTests(TestCase):
    def setUp(self):
        self.experiment = Experiment.objects.create(name="Exp1")
        self.run = TrainingRun.objects.create(experiment=self.experiment, job_type="training")

    def test_create_snapshot(self):
        snap = TrainingSnapshot.objects.create(
            run=self.run,
            step=10,
            train_loss=0.5,
            eval_accuracy=0.9,
            cpu_percent=55.2,
            ram_mb=1024.0
        )
        self.assertEqual(snap.step, 10)
        self.assertEqual(snap.train_loss, 0.5)
        self.assertIsNotNone(snap.timestamp)

    def test_unique_together(self):
        TrainingSnapshot.objects.create(run=self.run, step=1)
        with self.assertRaises(Exception):   # IntegrityError
            TrainingSnapshot.objects.create(run=self.run, step=1)


# ----------------------------------------------------------------------
# Serializer tests
# ----------------------------------------------------------------------
class SerializerTests(TestCase):
    def setUp(self):
        self.exp = Experiment.objects.create(name="Serializer Exp", tags="test")
        self.run = TrainingRun.objects.create(experiment=self.exp, name="Run", job_type="training")

    def test_experiment_serializer(self):
        data = ExperimentSerializer(self.exp).data
        self.assertEqual(data['name'], "Serializer Exp")
        self.assertIn('runs_count', data)

    def test_training_run_serializer(self):
        data = TrainingRunSerializer(self.run).data
        self.assertEqual(data['experiment_name'], "Serializer Exp")
        self.assertEqual(data['status'], "queued")

    def test_training_snapshot_serializer(self):
        snap = TrainingSnapshot.objects.create(run=self.run, step=5, train_loss=0.2)
        data = TrainingSnapshotSerializer(snap).data
        self.assertEqual(data['step'], 5)
        self.assertEqual(data['train_loss'], 0.2)


# ----------------------------------------------------------------------
# View tests (authenticated)
# ----------------------------------------------------------------------
class ExperimentViewTests(AuthMixin, APITestCase):
    def test_list_experiments(self):
        Experiment.objects.create(name="Exp1")
        Experiment.objects.create(name="Exp2")
        response = self.client.get(reverse('list_experiments'))  # you may need to set name in urls
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 2)

    def test_create_experiment(self):
        response = self.client.post('/api/experiments/create/', {
            'name': 'NewExp',
            'tags': 'demo'
        }, format='json')
        self.assertEqual(response.status_code, 201)
        self.assertEqual(Experiment.objects.count(), 1)
        self.assertEqual(Experiment.objects.first().name, 'NewExp')

    def test_experiment_detail(self):
        exp = Experiment.objects.create(name="Detail Exp")
        response = self.client.get(f'/api/experiments/{exp.id}/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['name'], "Detail Exp")


class TrainingRunViewTests(AuthMixin, APITestCase):
    def setUp(self):
        super().setUp()
        self.exp = Experiment.objects.create(name="RunTestExp")
        self.run = TrainingRun.objects.create(experiment=self.exp, name="TestRun", job_type="training")

    def test_list_runs(self):
        response = self.client.get('/api/runs/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)

    def test_run_detail(self):
        response = self.client.get(f'/api/runs/{self.run.id}/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['status'], 'queued')

    def test_start_train(self):
        data = {
            'experiment_id': str(self.exp.id),
            'name': 'Train1',
            'job_type': 'training',
            'config': {'epochs': 1}
        }
        response = self.client.post('/api/train/', data, format='json')
        self.assertEqual(response.status_code, 202)
        self.assertEqual(TrainingRun.objects.count(), 2)
        run = TrainingRun.objects.get(name='Train1')
        self.assertEqual(run.status, 'queued')   # start_job is async, status may still be queued; later it will change

    def test_start_evaluate_with_run_id_in_url(self):
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

    def test_training_snapshots(self):
        TrainingSnapshot.objects.create(run=self.run, step=1, train_loss=0.9)
        response = self.client.get(f'/api/runs/{self.run.id}/snapshots/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)

    def test_update_run_config(self):
        response = self.client.patch(f'/api/runs/{self.run.id}/config/',
                                     {'learning_rate': 0.001}, format='json')
        self.assertEqual(response.status_code, 200)
        self.run.refresh_from_db()
        self.assertEqual(self.run.config['learning_rate'], 0.001)

    def test_stop_all_jobs(self):
        response = self.client.post('/api/jobs/stop_all/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['status'], "stop signals sent to all running jobs")


class SystemViewTests(AuthMixin, APITestCase):
    def test_system_status(self):
        response = self.client.get('/api/system/status/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('cpu', response.data)

    def test_gpu_info(self):
        response = self.client.get('/api/system/gpu/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('type', response.data)


class AuthViewTests(APITestCase):
    def test_register(self):
        response = self.client.post('/api/auth/register/', {
            'username': 'newuser',
            'password': 'newpass123'
        })
        self.assertEqual(response.status_code, 201)
        self.assertIn('token', response.data)

    def test_login(self):
        User.objects.create_user(username='loginuser', password='loginpass')
        response = self.client.post('/api/auth/login/', {
            'username': 'loginuser',
            'password': 'loginpass'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('token', response.data)

    def test_logout(self):
        user = User.objects.create_user(username='logoutuser', password='logoutpass')
        token = Token.objects.create(user=user)
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + token.key)
        response = self.client.post('/api/auth/logout/')
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Token.objects.filter(user=user).exists())


# ----------------------------------------------------------------------
# Runner tests (direct, not via view)
# ----------------------------------------------------------------------
class RunnerTests(TestCase):
    def setUp(self):
        self.exp = Experiment.objects.create(name="RunnerExp")
        self.run = TrainingRun.objects.create(experiment=self.exp, job_type="training")
        self.stop_event = MagicMock()
        self.stop_event.is_set.return_value = False

    @patch('experiments.runners.push_update')   # silence channel messages
    @patch('experiments.runners._log')
    def test_run_training_completes(self, mock_log, mock_push):
        # run_training will use stubs that finish quickly
        run_training(self.run, self.stop_event)
        self.run.refresh_from_db()
        self.assertIn(self.run.status, ['completed', 'running'])   # stub might finish instantly
        # Check that final metrics were set
        self.assertIn('eval_accuracy', self.run.final_metrics)

    @patch('experiments.runners.push_update')
    @patch('experiments.runners._log')
    def test_run_training_stopped(self, mock_log, mock_push):
        self.stop_event.is_set.return_value = True
        run_training(self.run, self.stop_event)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, 'stopped')

    @patch('experiments.runners.push_update')
    @patch('experiments.runners._log')
    def test_run_evaluation(self, mock_log, mock_push):
        run_evaluation(self.run, self.stop_event)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, 'completed')
        self.assertIn('eval_accuracy', self.run.final_metrics)

    @patch('experiments.runners.push_update')
    @patch('experiments.runners._log')
    def test_run_compression(self, mock_log, mock_push):
        run_compression(self.run, self.stop_event)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, 'completed')
        self.assertIn('compression_ratio', self.run.final_metrics)


# ----------------------------------------------------------------------
# Consumer tests (WebSocket)
# ----------------------------------------------------------------------
class JobConsumerTests(TransactionTestCase):
    def setUp(self):
        # Create a run that will be used for WebSocket testing
        self.exp = Experiment.objects.create(name="WSEXP")
        self.run = TrainingRun.objects.create(experiment=self.exp, name="WSRun", job_type="training")

    def test_connect_to_job(self):
        """Verify WebSocket connection accepts and joins the group."""
        async def test():
            communicator = WebsocketCommunicator(application, f"/ws/jobs/{self.run.id}/")
            connected, _ = await communicator.connect()
            self.assertTrue(connected)
            await communicator.disconnect()
        async_to_sync(test)()

    def test_receive_job_update(self):
        """Broadcast a message to the job group and ensure consumer receives it."""
        async def test():
            communicator = WebsocketCommunicator(application, f"/ws/jobs/{self.run.id}/")
            connected, _ = await communicator.connect()
            self.assertTrue(connected)

            # Simulate a group_send to the job group
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
        """Broadcast to the terminal group and verify TerminalConsumer receives."""
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
            self.assertEqual(response, "Hello from test")

            await communicator.disconnect()
        async_to_sync(test)()