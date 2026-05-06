import json
from django.test import TestCase, Client, override_settings
from django.urls import reverse
from django.contrib.auth.models import User
from rest_framework.test import APIClient
from rest_framework.authtoken.models import Token

from experiment.models import Experiment, TrainingRun

# ----------------------------------------------------------------------
# Override middleware to remove the custom TokenAuthMiddleware during tests
# ----------------------------------------------------------------------
TEST_MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]


@override_settings(MIDDLEWARE=TEST_MIDDLEWARE)
class WebDashboardViewTests(TestCase):
    """Tests for all HTML page views and API endpoints in web_dashboard."""

    def setUp(self):
        self.client = Client()
        self.api_client = APIClient()

        self.user = User.objects.create_user(username='testuser', password='testpass')

        self.exp = Experiment.objects.create(name='Sample Experiment')
        self.run = TrainingRun.objects.create(
            experiment=self.exp,
            name='Sample Run',
            status='running',
            config={'model_name': 'bert', 'task_name': 'mrpc'},
            final_metrics={'eval_accuracy': 0.85},
            current_step=50,
            total_steps=100,
        )
        # A completed run with all required metrics for the statistics view
        TrainingRun.objects.create(
            experiment=self.exp,
            name='Completed Run',
            status='completed',
            final_metrics={'eval_accuracy': 0.9, 'eval_loss': 0.2, 'eval_f1': 0.88},
            config={'model_name': 'distilbert', 'task_name': 'sst2'},
        )

    # ------------------------------------------------------------------
    # Public pages (no login required)
    # ------------------------------------------------------------------
    def test_index(self):
        response = self.client.get(reverse('dashboard_index'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Transformer Pro')   # in the logo

    def test_create_job_page(self):
        response = self.client.get(reverse('dashboard_create_job'))
        self.assertEqual(response.status_code, 200)

    def test_login_page(self):
        response = self.client.get(reverse('dashboard_login'))
        self.assertEqual(response.status_code, 200)

    def test_register_page(self):
        response = self.client.get(reverse('dashboard_register'))
        self.assertEqual(response.status_code, 200)

    # ------------------------------------------------------------------
    # Authenticated pages
    # ------------------------------------------------------------------
    def test_job_detail_authenticated(self):
        self.client.force_login(self.user)
        url = reverse('dashboard_job_detail', kwargs={'job_id': self.run.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, str(self.run.id))   # injected in JS

    def test_dashboard_authenticated(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse('dashboard_dashboard'))
        self.assertEqual(response.status_code, 200)

    def test_statistics_authenticated(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse('dashboard_statistics'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'success_rate')   # a context variable rendered

    def test_job_detail_not_found_returns_404_when_authenticated(self):
        self.client.force_login(self.user)
        url = reverse('dashboard_job_detail', kwargs={
            'job_id': '00000000-0000-0000-0000-000000000000'
        })
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

    # ------------------------------------------------------------------
    # Redirect tests (not logged in)
    # ------------------------------------------------------------------
    def test_job_detail_redirect_if_not_logged_in(self):
        url = reverse('dashboard_job_detail', kwargs={'job_id': self.run.id})
        response = self.client.get(url)
        self.assertRedirects(response, f'/login/?next={url}')

    def test_dashboard_redirect_if_not_logged_in(self):
        url = reverse('dashboard_dashboard')
        response = self.client.get(url)
        self.assertRedirects(response, f'/login/?next={url}')

    def test_statistics_redirect_if_not_logged_in(self):
        url = reverse('dashboard_statistics')
        response = self.client.get(url)
        self.assertRedirects(response, f'/login/?next={url}')

    # ------------------------------------------------------------------
    # API endpoints (using APIClient)
    # ------------------------------------------------------------------
    def test_api_register_new_user(self):
        data = {'username': 'newuser', 'password': 'newpass123'}
        response = self.api_client.post(
            reverse('dashboard_api_register'),
            data=json.dumps(data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        self.assertIn('token', response.json())
        self.assertIn('authToken', response.cookies)

    def test_api_register_duplicate_user(self):
        User.objects.create_user(username='existing', password='pass')
        data = {'username': 'existing', 'password': 'pass'}
        response = self.api_client.post(
            reverse('dashboard_api_register'),
            data=json.dumps(data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn('Username already taken', response.json()['error'])

    def test_api_login_valid(self):
        User.objects.create_user(username='loginuser', password='loginpass')
        data = {'username': 'loginuser', 'password': 'loginpass'}
        response = self.api_client.post(
            reverse('dashboard_api_login'),
            data=json.dumps(data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn('token', json_data)
        self.assertIn('authToken', response.cookies)

    def test_api_login_invalid(self):
        response = self.api_client.post(
            reverse('dashboard_api_login'),
            data=json.dumps({'username': 'bad', 'password': 'wrong'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn('Invalid credentials', response.json()['error'])

    def test_api_logout(self):
        user = User.objects.create_user(username='logoutuser', password='logoutpass')
        token = Token.objects.create(user=user)

        # Simulate cookie-based authentication:
        # 1. Force login so the user is known (request.user)
        self.client.force_login(user)
        # 2. Set the authToken cookie so CookieTokenAuthentication finds the token
        self.client.cookies['authToken'] = token.key

        response = self.client.post(reverse('dashboard_api_logout'))
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Token.objects.filter(user=user).exists())
        self.assertIn('authToken', response.cookies)   # cookie cleared


    def test_dashboard_api_jobs(self):
        self.api_client.force_authenticate(user=self.user)
        response = self.api_client.get(reverse('dashboard_api_jobs'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        run = data[0]
        self.assertIn('id', run)
        self.assertIn('status', run)