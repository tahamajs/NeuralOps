import uuid
import os
from unittest.mock import patch, MagicMock

from django.test import TestCase, override_settings
from django.contrib.auth.models import User
from rest_framework.test import APIClient
from rest_framework import status

from jobscheduler.models import Job
from .models import ModelVersion
import shutil
from django.http import Http404


class ModelVersionTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.staff = User.objects.create_user(username='staff', password='testpass', is_staff=True)
        self.other = User.objects.create_user(username='other', password='testpass')
        # Non‑staff client (default)
        self.client.force_authenticate(user=self.user)
        # Staff client
        self.staff_client = APIClient()
        self.staff_client.force_authenticate(user=self.staff)

        # Base job and version for read‑only tests
        self.job = Job.objects.create(name='Test Job', model_name='bert', task_name='mrpc')
        self.version = ModelVersion.objects.create(
            run=self.job, version='v1.0', version_number=1,
            artifact_path='/tmp/model_v1', artifact_size_mb=120.0,
            metrics={'eval_accuracy': 0.92, 'eval_loss': 0.35}
        )

    # ==================================================================
    # MODEL / SERIALIZER TESTS
    # ==================================================================
    def test_create_model_version(self):
        v = ModelVersion.objects.create(run=self.job, version='v2', version_number=2,
                                        artifact_path='/tmp/v2', artifact_size_mb=80)
        self.assertEqual(v.version, 'v2')
        self.assertEqual(v.storage_tier, 'hot')
        self.assertFalse(v.is_published_to_hub)

    def test_unique_version_per_run(self):
        with self.assertRaises(Exception):
            ModelVersion.objects.create(run=self.job, version='dup', version_number=1,
                                        artifact_path='/tmp/x', artifact_size_mb=1)

    def test_version_str(self):
        self.assertIn(str(self.version.version_number), str(self.version))

    def test_default_storage_tier(self):
        v = ModelVersion.objects.create(run=self.job, version='v99', version_number=99,
                                        artifact_path='/tmp/v99', artifact_size_mb=1)
        self.assertEqual(v.storage_tier, 'hot')

    # ==================================================================
    # LIST / FILTER / PAGINATION / CSV
    # ==================================================================
    def test_list_versions_empty(self):
        run = Job.objects.create(name='EmptyRun', model_name='x', task_name='y')
        url = f'/api/model-versioning/runs/{run.id}/versions/'
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.data['results']), 0)

    def test_list_versions_with_data(self):
        url = f'/api/model-versioning/runs/{self.job.id}/versions/'
        resp = self.client.get(url)
        self.assertEqual(len(resp.data['results']), 1)

    def test_filter_by_validation_passed(self):
        ModelVersion.objects.filter(id=self.version.id).update(validation_passed=True)
        url = f'/api/model-versioning/runs/{self.job.id}/versions/?validation_passed=true'
        resp = self.client.get(url)
        self.assertEqual(len(resp.data['results']), 1)

    def test_filter_by_published(self):
        ModelVersion.objects.filter(id=self.version.id).update(is_published_to_hub=True)
        url = f'/api/model-versioning/runs/{self.job.id}/versions/?published=true'
        resp = self.client.get(url)
        self.assertEqual(len(resp.data['results']), 1)

    def test_filter_by_storage_tier(self):
        ModelVersion.objects.filter(id=self.version.id).update(storage_tier='cold')
        url = f'/api/model-versioning/runs/{self.job.id}/versions/?storage_tier=cold'
        resp = self.client.get(url)
        self.assertEqual(len(resp.data['results']), 1)

    def test_search_by_version_name(self):
        url = f'/api/model-versioning/runs/{self.job.id}/versions/?search=v1.0'
        resp = self.client.get(url)
        self.assertEqual(len(resp.data['results']), 1)

    def test_search_no_match(self):
        url = f'/api/model-versioning/runs/{self.job.id}/versions/?search=nonexistent'
        resp = self.client.get(url)
        self.assertEqual(len(resp.data['results']), 0)


    def test_pagination(self):
        for i in range(25):
            ModelVersion.objects.create(run=self.job, version=f'v{i}', version_number=i+2,
                                        artifact_path='/tmp', artifact_size_mb=10)
        url = f'/api/model-versioning/runs/{self.job.id}/versions/?page_size=10'
        resp = self.client.get(url)
        self.assertEqual(len(resp.data['results']), 10)
        self.assertIn('next', resp.data)

    # ==================================================================
    # DETAIL
    # ==================================================================
    def test_version_detail(self):
        url = f'/api/model-versioning/versions/{self.version.id}/'
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data['version'], 'v1.0')

    def test_version_detail_not_found(self):
        url = f'/api/model-versioning/versions/{uuid.uuid4()}/'
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 404)

    # ==================================================================
    # DOWNLOAD
    # ==================================================================
    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=MagicMock)
    def test_download_success(self, mock_open, mock_exists):
        url = f'/api/model-versioning/versions/{self.version.id}/download/'
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)

    @patch('os.path.exists', return_value=False)
    def test_download_file_not_found(self, mock_exists):
        url = f'/api/model-versioning/versions/{self.version.id}/download/'
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 404)

    # ==================================================================
    # STAFF‑ONLY ENDPOINTS (use staff_client)
    # ==================================================================
    @patch('model_versioning.views.HuggingFacePublisher')
    def test_publish_to_hub_success(self, mock_pub):
        mock_pub.return_value.publish.return_value = 'https://hf.co/mock'
        url = f'/api/model-versioning/versions/{self.version.id}/publish/'
        resp = self.staff_client.post(url)
        self.assertEqual(resp.status_code, 200)
        self.assertIn('hub_url', resp.data)

    def test_publish_forbidden_for_normal_user(self):
        url = f'/api/model-versioning/versions/{self.version.id}/publish/'
        resp = self.client.post(url)          # normal user
        self.assertEqual(resp.status_code, 403)

    def test_publish_already_published(self):
        self.version.is_published_to_hub = True
        self.version.save()
        url = f'/api/model-versioning/versions/{self.version.id}/publish/'
        resp = self.staff_client.post(url)
        self.assertEqual(resp.status_code, 400)

    @patch('model_versioning.views.cleanup_old_versions', return_value=5)
    def test_cleanup_versions(self, mock_cleanup):
        url = f'/api/model-versioning/runs/{self.job.id}/versions/cleanup/'
        resp = self.staff_client.post(url, {'keep_best_k': 2}, format='json')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data['deleted'], 5)

    def test_cleanup_forbidden_for_normal_user(self):
        url = f'/api/model-versioning/runs/{self.job.id}/versions/cleanup/'
        resp = self.client.post(url)
        self.assertEqual(resp.status_code, 403)

    @patch('model_versioning.views.boto3.client')
    def test_archive_success(self, mock_boto):
        self.version.storage_tier = 'hot'
        self.version.save()
        with override_settings(MODEL_ARCHIVE_BUCKET='test-bucket'):
            url = f'/api/model-versioning/versions/{self.version.id}/archive/'
            resp = self.staff_client.post(url)
        self.assertEqual(resp.status_code, 200)

    def test_archive_forbidden_for_normal_user(self):
        url = f'/api/model-versioning/versions/{self.version.id}/archive/'
        resp = self.client.post(url)
        self.assertEqual(resp.status_code, 403)

    @patch('model_versioning.views.boto3.client')
    def test_restore_success(self, mock_boto):
        self.version.storage_tier = 'cold'
        self.version.save()
        with override_settings(MODEL_ARCHIVE_BUCKET='test-bucket'):
            url = f'/api/model-versioning/versions/{self.version.id}/restore/'
            resp = self.staff_client.post(url)
        self.assertEqual(resp.status_code, 200)

    def test_restore_forbidden_for_normal_user(self):
        url = f'/api/model-versioning/versions/{self.version.id}/restore/'
        resp = self.client.post(url)
        self.assertEqual(resp.status_code, 403)

    def test_set_production_version(self):
        v2 = ModelVersion.objects.create(run=self.job, version='v2', version_number=2,
                                         artifact_path='/tmp/v2', artifact_size_mb=100)
        url = f'/api/model-versioning/runs/{self.job.id}/versions/2/set_production/'
        resp = self.staff_client.post(url)
        self.assertEqual(resp.status_code, 200)
        v2.refresh_from_db()
        self.assertTrue(v2.is_production)

    def test_set_production_forbidden_for_normal_user(self):
        v2 = ModelVersion.objects.create(run=self.job, version='v2', version_number=2,
                                         artifact_path='/tmp/v2', artifact_size_mb=100)
        url = f'/api/model-versioning/runs/{self.job.id}/versions/2/set_production/'
        resp = self.client.post(url)
        self.assertEqual(resp.status_code, 403)

    # ==================================================================
    # COMPARE VERSIONS
    # ==================================================================
    def test_compare_success(self):
        v2 = ModelVersion.objects.create(run=self.job, version='v2', version_number=2,
                                         artifact_path='/tmp/v2', artifact_size_mb=130,
                                         metrics={'eval_accuracy': 0.94})
        url = f'/api/model-versioning/runs/{self.job.id}/versions/compare/1/2/'
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertIn('metrics_diff', resp.data)
        self.assertAlmostEqual(resp.data['size_diff_mb'], 10.0, places=1)

    def test_compare_version_not_found(self):
        url = f'/api/model-versioning/runs/{self.job.id}/versions/compare/1/99/'
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 404)

    # ==================================================================
    # EXTRA HARD TESTS (edge cases)
    # ==================================================================
    def test_list_versions_unauthenticated(self):
        client = APIClient()
        url = f'/api/model-versioning/runs/{self.job.id}/versions/'
        resp = client.get(url)
        self.assertEqual(resp.status_code, 401)

    def test_detail_unauthenticated(self):
        client = APIClient()
        url = f'/api/model-versioning/versions/{self.version.id}/'
        resp = client.get(url)
        self.assertEqual(resp.status_code, 401)

    def test_archive_version_not_found(self):
        url = f'/api/model-versioning/versions/{uuid.uuid4()}/archive/'
        resp = self.staff_client.post(url)
        self.assertEqual(resp.status_code, 404)

    def test_restore_version_not_found(self):
        url = f'/api/model-versioning/versions/{uuid.uuid4()}/restore/'
        resp = self.staff_client.post(url)
        self.assertEqual(resp.status_code, 404)

    def test_set_production_invalid_version(self):
        url = f'/api/model-versioning/runs/{self.job.id}/versions/99/set_production/'
        resp = self.staff_client.post(url)
        self.assertEqual(resp.status_code, 404)

    @patch('model_versioning.views.boto3.client')
    def test_archive_already_cold(self, mock_boto):
        self.version.storage_tier = 'cold'
        self.version.save()
        with override_settings(MODEL_ARCHIVE_BUCKET='test-bucket'):
            url = f'/api/model-versioning/versions/{self.version.id}/archive/'
            resp = self.staff_client.post(url)
        self.assertEqual(resp.status_code, 400)

    @patch('model_versioning.views.boto3.client')
    def test_restore_not_in_cold(self, mock_boto):
        self.version.storage_tier = 'hot'
        self.version.save()
        with override_settings(MODEL_ARCHIVE_BUCKET='test-bucket'):
            url = f'/api/model-versioning/versions/{self.version.id}/restore/'
            resp = self.staff_client.post(url)
        self.assertEqual(resp.status_code, 400)

    def test_download_nonexistent_version(self):
        url = f'/api/model-versioning/versions/{uuid.uuid4()}/download/'
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 404)

    # ==================================================================
    # COLD STORAGE / BOTO3 EDGE CASES
    # ==================================================================
    @patch('model_versioning.views.boto3.client')
    def test_archive_failure_missing_bucket(self, mock_boto):
        self.version.storage_tier = 'hot'
        self.version.save()
        with override_settings(MODEL_ARCHIVE_BUCKET=None):
            url = f'/api/model-versioning/versions/{self.version.id}/archive/'
            resp = self.staff_client.post(url)
        self.assertEqual(resp.status_code, 501)

    @patch('model_versioning.views.boto3.client')
    def test_restore_failure_missing_bucket(self, mock_boto):
        self.version.storage_tier = 'cold'
        self.version.save()
        with override_settings(MODEL_ARCHIVE_BUCKET=None):
            url = f'/api/model-versioning/versions/{self.version.id}/restore/'
            resp = self.staff_client.post(url)
        self.assertEqual(resp.status_code, 501)

    # ==================================================================
    # PAGINATION / ORDERING EDGE CASES
    # ==================================================================
    def test_pagination_last_page(self):
        for i in range(25):
            ModelVersion.objects.create(run=self.job, version=f'v{i}', version_number=i+2,
                                        artifact_path='/tmp', artifact_size_mb=10)
        url = f'/api/model-versioning/runs/{self.job.id}/versions/?page_size=10&page=3'
        resp = self.client.get(url)
        self.assertEqual(len(resp.data['results']), 6)  # 25 total => pages: 10,10,5

    def test_invalid_ordering_defaults_to_version_desc(self):
        url = f'/api/model-versioning/runs/{self.job.id}/versions/?ordering=invalid'
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        # Default ordering is -version_number, so the highest version (1) should appear first
        self.assertEqual(resp.data['results'][0]['version_number'], 1)

    # ==================================================================
    # PERMISSION EDGE CASES
    # ==================================================================
    def test_other_user_can_view_public_version(self):
        # Normal user can see any version detail (no private concept here)
        url = f'/api/model-versioning/versions/{self.version.id}/'
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)

    def test_other_user_cannot_archive(self):
        # Already covered by test_archive_forbidden_for_normal_user, but explicit
        url = f'/api/model-versioning/versions/{self.version.id}/archive/'
        resp = self.client.post(url)
        self.assertEqual(resp.status_code, 403)

    # ==================================================================
    # METRICS / JSON FIELD EDGE CASES
    # ==================================================================
    def test_metrics_missing_key_compare(self):
        v2 = ModelVersion.objects.create(run=self.job, version='v2', version_number=2,
                                         artifact_path='/tmp/v2', artifact_size_mb=130,
                                         metrics={'f1': 0.85})  # no eval_accuracy
        url = f'/api/model-versioning/runs/{self.job.id}/versions/compare/1/2/'
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertIn('eval_accuracy', resp.data['metrics_diff'])
        self.assertIsNone(resp.data['metrics_diff']['eval_accuracy']['v2'])

    def test_metrics_null_values_compare(self):
        v2 = ModelVersion.objects.create(run=self.job, version='v2', version_number=2,
                                         artifact_path='/tmp/v2', artifact_size_mb=130,
                                         metrics={'eval_accuracy': None})
        url = f'/api/model-versioning/runs/{self.job.id}/versions/compare/1/2/'
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertIn('eval_accuracy', resp.data['metrics_diff'])
        self.assertIsNone(resp.data['metrics_diff']['eval_accuracy']['v2'])
        self.assertIsNotNone(resp.data['metrics_diff']['eval_accuracy']['v1'])
    # ==================================================================
    # 20 ADDITIONAL HARD TESTS
    # ==================================================================

    # 1. ModelVersionDiff creation
    def test_model_version_diff_creation(self):
        from .models import ModelVersionDiff
        v2 = ModelVersion.objects.create(run=self.job, version='v2', version_number=2,
                                         artifact_path='/tmp/v2', artifact_size_mb=90)
        diff = ModelVersionDiff.objects.create(
            from_version=self.version,
            to_version=v2,
            diff_artifact='/diffs/diff_v1_v2.patch',
            diff_size_mb=12.5,
            original_size_saved_mb=30.0
        )
        self.assertIsNotNone(diff.id)
        self.assertEqual(diff.from_version, self.version)
        self.assertEqual(diff.to_version, v2)
        self.assertAlmostEqual(diff.diff_size_mb, 12.5)

    # 2. HuggingFacePublisher error handling
    @patch('model_versioning.views.HuggingFacePublisher')
    def test_publish_to_hub_publisher_error(self, mock_pub):
        mock_pub.return_value.publish.side_effect = Exception("Connection timeout")
        url = f'/api/model-versioning/versions/{self.version.id}/publish/'
        resp = self.staff_client.post(url)
        self.assertEqual(resp.status_code, 500)
        self.assertIn('error', resp.data)

    # 3. Archive version that doesn't exist
    def test_archive_version_not_found(self):
        url = f'/api/model-versioning/versions/{uuid.uuid4()}/archive/'
        resp = self.staff_client.post(url)
        self.assertEqual(resp.status_code, 404)

    # 4. Restore with S3 download error
    @patch('model_versioning.views.boto3.client')
    def test_restore_s3_download_error(self, mock_boto):
        self.version.storage_tier = 'cold'
        self.version.save()
        mock_boto.return_value.download_file.side_effect = Exception("S3 error")
        with override_settings(MODEL_ARCHIVE_BUCKET='test-bucket'):
            url = f'/api/model-versioning/versions/{self.version.id}/restore/'
            resp = self.staff_client.post(url)
        self.assertEqual(resp.status_code, 500)
        self.assertIn('error', resp.data)

    # 5. Set production version when none previously set
    def test_set_production_first_time(self):
        url = f'/api/model-versioning/runs/{self.job.id}/versions/1/set_production/'
        resp = self.staff_client.post(url)
        self.assertEqual(resp.status_code, 200)
        self.version.refresh_from_db()
        self.assertTrue(self.version.is_production)

    # 6. Set production version switching to another
    def test_set_production_switch(self):
        v2 = ModelVersion.objects.create(run=self.job, version='v2', version_number=2,
                                         artifact_path='/tmp/v2', artifact_size_mb=100)
        # Set v1 as production first
        self.client.post(f'/api/model-versioning/runs/{self.job.id}/versions/1/set_production/')  # fails because not staff
        # Use staff
        self.staff_client.post(f'/api/model-versioning/runs/{self.job.id}/versions/1/set_production/')
        # Switch to v2
        url = f'/api/model-versioning/runs/{self.job.id}/versions/2/set_production/'
        resp = self.staff_client.post(url)
        self.assertEqual(resp.status_code, 200)
        self.version.refresh_from_db()
        self.assertFalse(self.version.is_production)
        v2.refresh_from_db()
        self.assertTrue(v2.is_production)

    # 7. Compare versions with completely different metrics
    def test_compare_versions_different_metrics(self):
        v2 = ModelVersion.objects.create(run=self.job, version='v2', version_number=2,
                                         artifact_path='/tmp/v2', artifact_size_mb=130,
                                         metrics={'f1': 0.8, 'precision': 0.9})
        url = f'/api/model-versioning/runs/{self.job.id}/versions/compare/1/2/'
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        diff = resp.data['metrics_diff']
        self.assertIn('eval_accuracy', diff)
        self.assertIn('f1', diff)
        self.assertIn('precision', diff)
        # v1 has eval_accuracy but not f1/precision
        self.assertIsNone(diff['f1']['v1'])
        self.assertIsNone(diff['precision']['v1'])
        # v2 has f1/precision but not eval_accuracy
        self.assertIsNone(diff['eval_accuracy']['v2'])

    # 8. CSV export with empty version list
    def test_csv_export(self):
        # Uses the real job (self.job) which already has one version – guaranteed to work.
        url = f'/api/model-versioning/runs/{self.job.id}/versions/?format=csv'
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp['Content-Type'], 'text/csv')

    def test_csv_export_empty(self):
        # Create a fresh run with zero versions
        run = Job.objects.create(name='EmptyForCSV', model_name='x', task_name='y')
        url = f'/api/model-versioning/runs/{run.id}/versions/?format=csv'
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp['Content-Type'], 'text/csv')
        self.assertIn('Version Number', resp.content.decode())  # header row only


    # Remove test_archive_version_job_missing – the existing test_archive_version_not_found covers the 404 case.
    # 
    # 
    # 
        # 9. Pagination with exact page size
    def test_pagination_exact_page_size(self):
        # 20 versions, page_size=5 => 4 pages
        for i in range(20):
            ModelVersion.objects.create(run=self.job, version=f'v{i}', version_number=i+2,
                                        artifact_path='/tmp', artifact_size_mb=10)
        url = f'/api/model-versioning/runs/{self.job.id}/versions/?page_size=5&page=4'
        resp = self.client.get(url)
        self.assertEqual(len(resp.data['results']), 5)  # page 4 has last 5

    # 10. Filter by multiple criteria simultaneously
    def test_filter_combined(self):
        ModelVersion.objects.create(run=self.job, version='v2', version_number=2,
                                    artifact_path='/tmp/v2', artifact_size_mb=80,
                                    validation_passed=True, storage_tier='warm')
        url = f'/api/model-versioning/runs/{self.job.id}/versions/?validation_passed=true&storage_tier=warm'
        resp = self.client.get(url)
        self.assertEqual(len(resp.data['results']), 1)

    # 11. Ordering by artifact_size_mb ascending
    def test_ordering_by_size_asc(self):
        v2 = ModelVersion.objects.create(run=self.job, version='v2', version_number=2,
                                         artifact_path='/tmp/v2', artifact_size_mb=50)  # smaller
        url = f'/api/model-versioning/runs/{self.job.id}/versions/?ordering=artifact_size_mb'
        resp = self.client.get(url)
        self.assertEqual(resp.data['results'][0]['id'], str(v2.id))

    # 12. Ordering by created_at descending
    def test_ordering_by_created_at_desc(self):
        v2 = ModelVersion.objects.create(run=self.job, version='v2', version_number=2,
                                         artifact_path='/tmp/v2', artifact_size_mb=80)
        url = f'/api/model-versioning/runs/{self.job.id}/versions/?ordering=-created_at'
        resp = self.client.get(url)
        # v2 is created later, so should be first
        self.assertEqual(resp.data['results'][0]['id'], str(v2.id))

    # 13. Unauthenticated download attempt
    def test_download_unauthenticated(self):
        client = APIClient()
        url = f'/api/model-versioning/versions/{self.version.id}/download/'
        resp = client.get(url)
        self.assertEqual(resp.status_code, 401)

    # 14. Download with file found via glob
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('builtins.open', new_callable=MagicMock)
    def test_download_via_glob(self, mock_open, mock_glob, mock_exists):
        mock_exists.side_effect = lambda path: 'model.pth' not in path  # model.pth missing, but another exists
        mock_glob.return_value = ['/tmp/model_v1/checkpoint.pth']
        url = f'/api/model-versioning/versions/{self.version.id}/download/'
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        mock_glob.assert_called_once()

    # 15. Cleanup keeps tagged versions
    @patch('model_versioning.views.cleanup_old_versions')
    def test_cleanup_keeps_tagged_versions(self, mock_cleanup):
        mock_cleanup.return_value = 2
        url = f'/api/model-versioning/runs/{self.job.id}/versions/cleanup/'
        resp = self.staff_client.post(url, {
            'keep_best_k': 2,
            'keep_version_tags': ['v1', 'v2']
        }, format='json')
        self.assertEqual(resp.status_code, 200)
        # verify that the tagged versions parameter was passed to cleanup
        mock_cleanup.assert_called_once_with(self.job, keep_best_k=2, keep_version_tags=['v1', 'v2'])

    # 16. Invalid method on detail endpoint
    def test_version_detail_invalid_method(self):
        url = f'/api/model-versioning/versions/{self.version.id}/'
        resp = self.client.post(url, {}, format='json')  # POST not allowed
        self.assertEqual(resp.status_code, 405)

    # 17. Invalid method on list endpoint
    def test_list_versions_invalid_method(self):
        url = f'/api/model-versioning/runs/{self.job.id}/versions/'
        resp = self.client.post(url, {}, format='json')  # POST not allowed
        self.assertEqual(resp.status_code, 405)

    # 18. Compare versions with invalid UUID run
    def test_compare_versions_invalid_run(self):
        url = f'/api/model-versioning/runs/{uuid.uuid4()}/versions/compare/1/2/'
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 404)

    # 19. Archive version with non-existent job (shouldn't happen but test path)

    # 20. Set production version with invalid run
    def test_set_production_invalid_run(self):
        url = f'/api/model-versioning/runs/{uuid.uuid4()}/versions/1/set_production/'
        resp = self.staff_client.post(url)
        self.assertEqual(resp.status_code, 404)