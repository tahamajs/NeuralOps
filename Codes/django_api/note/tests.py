# jobscheduler/tests/test_jobscheduler.py
import uuid
import json
import tempfile
from datetime import datetime
from django.test import TestCase
from django.contrib.auth.models import User
from django.utils import timezone
from rest_framework.test import APIClient
from unittest.mock import patch, MagicMock
from ..models import Job, CustomMetricEntry, VersionedConfig
from ..scheduler import GPUScheduler
from ..logger import CustomRunLogger

class JobModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpass')
    
    def test_create_job(self):
        job = Job.objects.create(
            name='Test Job',
            model_name='bert-base',
            task_name='mrpc',
            priority=3,
            author=self.user
        )
        self.assertEqual(job.status, 'queued')
        self.assertEqual(job.priority, 3)
        self.assertIsNotNone(job.created_at)
    
    def test_job_status_update(self):
        job = Job.objects.create(name='Test')
        job.status = 'running'
        job.save()
        self.assertEqual(job.status, 'running')
        self.assertIsNotNone(job.started_at)

class CustomMetricEntryTests(TestCase):
    def setUp(self):
        self.job = Job.objects.create(name='Metric Test Job')
    
    def test_create_scalar_metric(self):
        metric = CustomMetricEntry.objects.create(
            run=self.job,
            name='train_loss',
            type='scalar',
            value=0.234,
            step=100
        )
        self.assertEqual(metric.value, 0.234)
        self.assertEqual(metric.step, 100)
    
    def test_create_table_metric(self):
        table_data = [{'class': 'A', 'acc': 0.9}, {'class': 'B', 'acc': 0.85}]
        metric = CustomMetricEntry.objects.create(
            run=self.job,
            name='per_class_acc',
            type='table',
            value=table_data,
            step=200
        )
        self.assertEqual(len(metric.value), 2)

class APITests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='api_user', password='api123')
        self.client.force_authenticate(user=self.user)
    
    def test_job_list(self):
        Job.objects.create(name='Job1', model_name='bert', task_name='mrpc')
        Job.objects.create(name='Job2', model_name='roberta', task_name='sst2')
        response = self.client.get('/api/jobs/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 2)
    
    def test_create_job_api(self):
        data = {
            'name': 'API Job',
            'model_name': 'distilbert',
            'task_name': 'cola',
            'priority': 2
        }
        response = self.client.post('/api/jobs/', data, format='json')
        self.assertEqual(response.status_code, 201)
        self.assertEqual(Job.objects.count(), 1)
        self.assertEqual(response.data['name'], 'API Job')
    
    def test_update_job_priority(self):
        job = Job.objects.create(name='Priority Job', priority=5)
        response = self.client.patch(f'/api/jobs/{job.id}/priority/', {'priority': 2}, format='json')
        self.assertEqual(response.status_code, 200)
        job.refresh_from_db()
        self.assertEqual(job.priority, 2)
    
    def test_delete_job(self):
        job = Job.objects.create(name='ToDelete', status='completed')
        response = self.client.delete(f'/api/jobs/{job.id}/')
        self.assertEqual(response.status_code, 204)
        self.assertFalse(Job.objects.filter(id=job.id).exists())
    
    def test_custom_metrics_logging(self):
        job = Job.objects.create(name='Metrics Job')
        data = {
            'name': 'test_metric',
            'type': 'scalar',
            'value': 0.999,
            'step': 1
        }
        response = self.client.post(f'/api/runs/{job.id}/metrics/', data, format='json')
        self.assertEqual(response.status_code, 201)
        self.assertEqual(CustomMetricEntry.objects.count(), 1)
    
    def test_get_custom_metrics(self):
        job = Job.objects.create(name='GetMetrics')
        CustomMetricEntry.objects.create(run=job, name='loss', type='scalar', value=0.5, step=1)
        CustomMetricEntry.objects.create(run=job, name='acc', type='scalar', value=0.8, step=2)
        response = self.client.get(f'/api/runs/{job.id}/metrics/list/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('loss', response.data)
        self.assertIn('acc', response.data)

class SchedulerTests(TestCase):
    @patch('jobscheduler.scheduler.subprocess.Popen')
    def test_add_job_to_queue(self, mock_popen):
        scheduler = GPUScheduler(total_gpus=2)
        job = Job.objects.create(name='Sched Job', priority=3)
        mock_popen.return_value.wait.return_value = 0
        # Not testing actual process launch; just queue addition
        scheduler.add_job(str(job.id), job.priority)
        self.assertEqual(scheduler.task_queue.qsize(), 1)
    
    def test_scheduler_singleton(self):
        s1 = GPUScheduler(total_gpus=1)
        s2 = GPUScheduler()
        self.assertIs(s1, s2)

class LoggerTests(TestCase):
    def setUp(self):
        self.job = Job.objects.create(name='Logger Test Job')
    
    @patch('jobscheduler.logger.requests.Session.post')
    def test_log_scalar(self, mock_post):
        mock_post.return_value.status_code = 201
        logger = CustomRunLogger(str(self.job.id), async_mode=False)
        logger.log_scalar('train_loss', 0.123, step=5)
        self.assertTrue(mock_post.called)
        call_args = mock_post.call_args[1]['json']
        self.assertEqual(call_args['name'], 'train_loss')
        self.assertEqual(call_args['value'], 0.123)
    
    @patch('jobscheduler.logger.default_storage.save')
    def test_log_image_with_upload(self, mock_save):
        mock_save.return_value = 'metrics/test.png'
        logger = CustomRunLogger(str(self.job.id), async_mode=False, upload_images=True)
        from PIL import Image
        img = Image.new('RGB', (10, 10))
        logger.log_image('test_img', img, step=1)
        self.assertTrue(mock_save.called)