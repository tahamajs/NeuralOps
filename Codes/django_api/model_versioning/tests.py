import uuid
import json
import os
from unittest.mock import patch, MagicMock, PropertyMock

from django.test import TestCase, override_settings
from django.contrib.auth.models import User
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status

from jobscheduler.models import Job
from .models import ModelVersion, ModelVersionDiff
from .serializers import ModelVersionSerializer, ModelVersionDiffSerializer


def _create_job_with_author(user):
    job = Job.objects.create(name='Test Run', model_name='bert', task_name='mrpc')
    job.author = user
    return job


class ModelVersionTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.staff_user = User.objects.create_user(username='staff', password='testpass', is_staff=True)
        # default client is authenticated as normal user
        self.client.force_authenticate(user=self.user)

        # real job and version for read-only endpoints
        self.job = Job.objects.create(name='Run 1', model_name='bert', task_name='mrpc')
        self.version = ModelVersion.objects.create(
            run=self.job,
            version='v1.0',
            version_number=1,
            artifact_path='/tmp/test_artifact',
            artifact_size_mb=120.5,
            metrics={'eval_accuracy': 0.92, 'eval_loss': 0.35},
            is_production=False,
            storage_tier='hot'
        )

        # mock jobs for permission tests
        self.mock_job = _create_job_with_author(self.user)
        self.mock_job_other = _create_job_with_author(User.objects.create_user(username='other', password='x'))

        # a staff-authenticated client for restricted endpoints
        self.staff_client = APIClient()
        self.staff_client.force_authenticate(user=self.staff_user)

    # ==================================================================
    # MODEL / SERIALIZER TESTS (unchanged)
    # ==================================================================
    def test_model_version_creation(self):
        v = ModelVersion.objects.create(
            run=self.job, version='v2.0', version_number=2,
            artifact_path='/tmp/v2', artifact_size_mb=80.0
        )
        self.assertIsNotNone(v.id)
        self.assertEqual(v.version, 'v2.0')
        self.assertEqual(v.storage_tier, 'hot')
        self.assertFalse(v.is_published_to_hub)

    def test_unique_together_run_version_number(self):
        ModelVersion.objects.create(run=self.job, version='v3', version_number=3, artifact_path='/tmp/x', artifact_size_mb=1)
        with self.assertRaises(Exception):
            ModelVersion.objects.create(run=self.job, version='v3_dup', version_number=3, artifact_path='/tmp/y', artifact_size_mb=1)

    def test_str_method(self):
        self.assertIn(str(self.version.version_number), str(self.version))

    def test_model_version_serializer(self):
        serializer = ModelVersionSerializer(self.version)
        data = serializer.data
        self.assertEqual(data['version'], 'v1.0')
        self.assertEqual(data['version_number'], 1)
        self.assertIn('metrics', data)

    # ==================================================================
    # LIST ENDPOINTS
    # ==================================================================
    @patch('model_versioning.views.get_object_or_404')
    def test_list_model_versions(self, mock_get_object):
        mock_get_object.return_value = self.mock_job
        ModelVersion.objects.create(run=self.mock_job, version='v1', version_number=1, artifact_path='/tmp/a', artifact_size_mb=10)
        ModelVersion.objects.create(run=self.mock_job, version='v2', version_number=2, artifact_path='/tmp/b', artifact_size_mb=20)
        url = f'/api/model-versioning/runs/{self.mock_job.id}/versions/'
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertIn('results', response.data)

    def test_list_versions_filter_validation(self):
        ModelVersion.objects.filter(id=self.version.id).update(validation_passed=True)
        url = f'/api/model-versioning/runs/{self.job.id}/versions/?validation_passed=true'
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data['results']), 1)

    @patch('model_versioning.views.get_object_or_404')
    def test_list_versions_csv_export(self, mock_get_object):
        # create a mock job with a model_versions manager that returns a queryset
        mock_run = MagicMock()
        mock_run.id = self.mock_job.id
        mock_manager = MagicMock()
        mock_manager.all.return_value.order_by.return_value = ModelVersion.objects.filter(id=self.version.id)
        mock_run.model_versions = mock_manager
        mock_get_object.return_value = mock_run
        url = f'/api/model-versioning/runs/{self.mock_job.id}/versions/?format=csv'
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/csv')

    # ==================================================================
    # DETAIL
    # ==================================================================
    def test_model_version_detail(self):
        url = f'/api/model-versioning/versions/{self.version.id}/'
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['version'], 'v1.0')

    def test_model_version_detail_not_found(self):
        url = f'/api/model-versioning/versions/{uuid.uuid4()}/'
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

    # ==================================================================
    # PUBLISH (requires staff)
    # ==================================================================
    @patch('model_versioning.views.HuggingFacePublisher')
    @patch('model_versioning.views.get_object_or_404')
    def test_publish_to_hub_success(self, mock_get_object, mock_publisher_cls):
        mock_version = self.version
        mock_version.run = self.mock_job
        mock_get_object.return_value = mock_version
        mock_publisher = mock_publisher_cls.return_value
        mock_publisher.publish.return_value = 'https://huggingface.co/mock/model'

        url = f'/api/model-versioning/versions/{mock_version.id}/publish/'
        response = self.staff_client.post(url)                     # staff client
        self.assertEqual(response.status_code, 200)
        self.assertIn('hub_url', response.data)

    @patch('model_versioning.views.HuggingFacePublisher')
    @patch('model_versioning.views.get_object_or_404')
    def test_publish_already_published(self, mock_get_object, mock_publisher_cls):
        mock_version = self.version
        mock_version.run = self.mock_job
        mock_version.is_published_to_hub = True
        mock_get_object.return_value = mock_version
        url = f'/api/model-versioning/versions/{mock_version.id}/publish/'
        response = self.staff_client.post(url)
        self.assertEqual(response.status_code, 400)

    @patch('model_versioning.views.get_object_or_404')
    def test_publish_permission_denied(self, mock_get_object):
        mock_version = self.version
        mock_version.run = self.mock_job_other
        mock_get_object.return_value = mock_version
        url = f'/api/model-versioning/versions/{mock_version.id}/publish/'
        response = self.client.post(url)          # normal user, should be forbidden
        self.assertEqual(response.status_code, 403)

    # ==================================================================
    # DOWNLOAD
    # ==================================================================
    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=MagicMock)
    def test_download_model_version(self, mock_open, mock_exists):
        url = f'/api/model-versioning/versions/{self.version.id}/download/'
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)

    @patch('os.path.exists', return_value=False)
    def test_download_model_not_found(self, mock_exists):
        url = f'/api/model-versioning/versions/{self.version.id}/download/'
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

    # ==================================================================
    # CLEANUP (requires staff)
    # ==================================================================
    @patch('model_versioning.views.cleanup_old_versions')
    @patch('model_versioning.views.get_object_or_404')
    def test_cleanup_versions(self, mock_get_object, mock_cleanup):
        mock_get_object.return_value = self.mock_job
        mock_cleanup.return_value = 5
        url = f'/api/model-versioning/runs/{self.mock_job.id}/versions/cleanup/'
        response = self.staff_client.post(url, {'keep_best_k': 2}, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['deleted'], 5)

    @patch('model_versioning.views.get_object_or_404')
    def test_cleanup_permission_denied(self, mock_get_object):
        mock_get_object.return_value = self.mock_job_other
        url = f'/api/model-versioning/runs/{self.mock_job_other.id}/versions/cleanup/'
        response = self.client.post(url, {'keep_best_k': 2}, format='json')
        self.assertEqual(response.status_code, 403)

    # ==================================================================
    # ARCHIVE / RESTORE (requires staff, boto3 now at module level)
    # ==================================================================
    @patch('model_versioning.views.boto3.client')          # now works because boto3 is imported at top
    @patch('model_versioning.views.get_object_or_404')
    def test_archive_to_cold_storage(self, mock_get_object, mock_boto_client):
        mock_version = self.version
        mock_version.run = self.mock_job
        mock_version.storage_tier = 'hot'
        mock_get_object.return_value = mock_version
        with override_settings(MODEL_ARCHIVE_BUCKET='test-bucket'):
            url = f'/api/model-versioning/versions/{mock_version.id}/archive/'
            response = self.staff_client.post(url)
        self.assertEqual(response.status_code, 200)

    @patch('model_versioning.views.boto3.client')
    @patch('model_versioning.views.get_object_or_404')
    def test_restore_from_cold_storage(self, mock_get_object, mock_boto_client):
        mock_version = self.version
        mock_version.run = self.mock_job
        mock_version.storage_tier = 'cold'
        mock_get_object.return_value = mock_version
        with override_settings(MODEL_ARCHIVE_BUCKET='test-bucket'):
            url = f'/api/model-versioning/versions/{mock_version.id}/restore/'
            response = self.staff_client.post(url)
        self.assertEqual(response.status_code, 200)

    # ==================================================================
    # SET PRODUCTION VERSION (requires staff)
    # ==================================================================
    def test_set_production_version(self):
        v2 = ModelVersion.objects.create(
            run=self.job, version='v2', version_number=2,
            artifact_path='/tmp/v2', artifact_size_mb=100
        )
        url = f'/api/model-versioning/runs/{self.job.id}/versions/2/set_production/'
        response = self.staff_client.post(url)
        self.assertEqual(response.status_code, 200)

    # ==================================================================
    # COMPARE VERSIONS – use a fresh run to avoid unique constraint
    # ==================================================================
    def test_compare_versions(self):
        run = Job.objects.create(name='CompareRun', model_name='bert', task_name='sst2')
        v1 = ModelVersion.objects.create(run=run, version_number=1, version='v1', artifact_path='/tmp/v1',
                                         artifact_size_mb=10, metrics={'eval_accuracy': 0.91})
        v2 = ModelVersion.objects.create(run=run, version_number=2, version='v2', artifact_path='/tmp/v2',
                                         artifact_size_mb=11, metrics={'eval_accuracy': 0.93})
        url = f'/api/model-versioning/runs/{run.id}/versions/compare/1/2/'
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertIn('metrics_diff', response.data)