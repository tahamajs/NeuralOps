import json

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from experiment.models import Experiment, TrainingRun
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient

# ----------------------------------------------------------------------
# Override middleware to remove the custom TokenAuthMiddleware during tests
# ----------------------------------------------------------------------
TEST_MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]


@override_settings(MIDDLEWARE=TEST_MIDDLEWARE)
class WebDashboardViewTests(TestCase):
    """Tests for all HTML page views and API endpoints in web_dashboard."""

    def setUp(self):
        self.client = Client()
        self.api_client = APIClient()

        self.user = User.objects.create_user(username="testuser", password="testpass")

        self.exp = Experiment.objects.create(name="Sample Experiment")
        self.run = TrainingRun.objects.create(
            experiment=self.exp,
            name="Sample Run",
            status="running",
            config={"model_name": "bert", "task_name": "mrpc"},
            final_metrics={"eval_accuracy": 0.85},
            current_step=50,
            total_steps=100,
        )
        # A completed run with all required metrics for the statistics view
        TrainingRun.objects.create(
            experiment=self.exp,
            name="Completed Run",
            status="completed",
            final_metrics={"eval_accuracy": 0.9, "eval_loss": 0.2, "eval_f1": 0.88},
            config={"model_name": "distilbert", "task_name": "sst2"},
        )

    # ------------------------------------------------------------------
    # Public pages (no login required)
    # ------------------------------------------------------------------
    def test_index(self):
        response = self.client.get(reverse("dashboard_index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Transformer Pro")  # in the logo

    def test_create_job_page(self):
        response = self.client.get(reverse("dashboard_create_job"))
        self.assertEqual(response.status_code, 200)

    def test_login_page(self):
        response = self.client.get(reverse("dashboard_login"))
        self.assertEqual(response.status_code, 200)

    def test_register_page(self):
        response = self.client.get(reverse("dashboard_register"))
        self.assertEqual(response.status_code, 200)

    # ------------------------------------------------------------------
    # Authenticated pages
    # ------------------------------------------------------------------
    def test_job_detail_authenticated(self):
        self.client.force_login(self.user)
        url = reverse("dashboard_job_detail", kwargs={"job_id": self.run.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, str(self.run.id))  # injected in JS

    def test_dashboard_authenticated(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("dashboard_dashboard"))
        self.assertEqual(response.status_code, 200)

    def test_statistics_authenticated(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("dashboard_statistics"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "success_rate")  # a context variable rendered

    def test_job_detail_not_found_returns_404_when_authenticated(self):
        self.client.force_login(self.user)
        url = reverse(
            "dashboard_job_detail",
            kwargs={"job_id": "00000000-0000-0000-0000-000000000000"},
        )
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

    # ------------------------------------------------------------------
    # Redirect tests (not logged in)
    # ------------------------------------------------------------------
    def test_job_detail_redirect_if_not_logged_in(self):
        url = reverse("dashboard_job_detail", kwargs={"job_id": self.run.id})
        response = self.client.get(url)
        self.assertRedirects(response, f"/login/?next={url}", fetch_redirect_response=False)

    def test_dashboard_redirect_if_not_logged_in(self):
        url = reverse("dashboard_dashboard")
        response = self.client.get(url)
        self.assertRedirects(response, f"/login/?next={url}", fetch_redirect_response=False)

    def test_statistics_redirect_if_not_logged_in(self):
        url = reverse("dashboard_statistics")
        response = self.client.get(url)
        self.assertRedirects(response, f"/login/?next={url}", fetch_redirect_response=False)

    # ------------------------------------------------------------------
    # API endpoints (using APIClient)
    # ------------------------------------------------------------------
    def test_api_register_new_user(self):
        data = {"username": "newuser", "password": "newpass123"}
        response = self.api_client.post(
            reverse("dashboard_api_register"),
            data=json.dumps(data),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 201)
        self.assertIn("token", response.json())
        self.assertIn("authToken", response.cookies)

    def test_api_register_duplicate_user(self):
        User.objects.create_user(username="existing", password="pass")
        data = {"username": "existing", "password": "pass"}
        response = self.api_client.post(
            reverse("dashboard_api_register"),
            data=json.dumps(data),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Username already taken", response.json()["error"])

    def test_api_login_valid(self):
        User.objects.create_user(username="loginuser", password="loginpass")
        data = {"username": "loginuser", "password": "loginpass"}
        response = self.api_client.post(
            reverse("dashboard_api_login"),
            data=json.dumps(data),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("token", json_data)
        self.assertIn("authToken", response.cookies)

    def test_api_login_invalid(self):
        response = self.api_client.post(
            reverse("dashboard_api_login"),
            data=json.dumps({"username": "bad", "password": "wrong"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn("Invalid credentials", response.json()["error"])

    def test_api_logout(self):
        user = User.objects.create_user(username="logoutuser", password="logoutpass")
        token = Token.objects.create(user=user)

        # Simulate cookie-based authentication:
        # 1. Force login so the user is known (request.user)
        self.client.force_login(user)
        # 2. Set the authToken cookie so CookieTokenAuthentication finds the token
        self.client.cookies["authToken"] = token.key

        response = self.client.post(reverse("dashboard_api_logout"))
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Token.objects.filter(user=user).exists())
        self.assertIn("authToken", response.cookies)  # cookie cleared

    def test_dashboard_api_jobs(self):
        self.api_client.force_authenticate(user=self.user)
        response = self.api_client.get(reverse("dashboard_api_jobs"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        run = data[0]
        self.assertIn("id", run)
        self.assertIn("status", run)

        # ------------------------------------------------------------------

    # Additional hard tests (20)
    # ------------------------------------------------------------------

    # 1. Index with no runs (empty dataset)
    def test_index_empty_runs(self):
        TrainingRun.objects.all().delete()
        response = self.client.get(reverse("dashboard_index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Transformer Pro")  # page loads anyway

    # 2. Job detail with invalid UUID format
    def test_job_detail_invalid_uuid(self):
        self.client.force_login(self.user)
        # Manually construct a path with a non-UUID value – Django will return 404
        response = self.client.get("/dashboard/job/not-a-uuid/")
        self.assertEqual(response.status_code, 404)

    # 3. Statistics with no runs (empty data)
    def test_statistics_empty(self):
        self.client.force_login(self.user)
        TrainingRun.objects.all().delete()
        response = self.client.get(reverse("dashboard_statistics"))
        self.assertEqual(response.status_code, 200)
        # Our minimal template renders: success_rate: 0  (float 0.0 is rendered as integer 0 by Django?)
        # Actually 0.0 renders as '0.0' but we see '0' in the response. To be safe we check for 'success_rate: 0'
        self.assertContains(response, "success_rate: 0")  # no decimal

    # 4. API register with missing username
    def test_api_register_missing_username(self):
        data = {"password": "test123"}
        response = self.api_client.post(
            reverse("dashboard_api_register"),
            data=json.dumps(data),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    # 5. API register with missing password
    def test_api_register_missing_password(self):
        data = {"username": "user123"}
        response = self.api_client.post(
            reverse("dashboard_api_register"),
            data=json.dumps(data),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    # 6. API login with missing fields
    def test_api_login_missing_fields(self):
        response = self.api_client.post(
            reverse("dashboard_api_login"),
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn("error", response.json())

    # 7. API logout when not authenticated
    def test_api_logout_unauthenticated(self):
        response = self.client.post(reverse("dashboard_api_logout"))
        self.assertEqual(response.status_code, 401)  # or 403; depends on DRF default

    # 8. API jobs endpoint unauthenticated
    def test_dashboard_api_jobs_unauthenticated(self):
        response = self.api_client.get(reverse("dashboard_api_jobs"))
        self.assertEqual(response.status_code, 401)

    # 9. API register with very long username/password (boundary)
    def test_api_register_long_credentials(self):
        long_user = "a" * 151  # Django's default max_length is 150
        data = {"username": long_user, "password": "validpass"}
        response = self.api_client.post(
            reverse("dashboard_api_register"),
            data=json.dumps(data),
            content_type="application/json",
        )
        # Should fail because username too long (400)
        self.assertEqual(response.status_code, 400)

    # 10. Statistics when a run has missing final_metrics keys
    def test_statistics_partial_metrics(self):
        # Create a completed run without 'eval_loss' key
        TrainingRun.objects.create(
            experiment=self.exp,
            name="Partial Metrics Run",
            status="completed",
            final_metrics={"eval_accuracy": 0.5},  # missing eval_loss
            config={"model_name": "xlm"},
        )
        self.client.force_login(self.user)
        response = self.client.get(reverse("dashboard_statistics"))
        self.assertEqual(response.status_code, 200)
        # View should not crash (uses Coalesce)
        self.assertContains(response, "success_rate")

    # 11. Dashboard API jobs with empty list
    def test_dashboard_api_jobs_empty(self):
        TrainingRun.objects.all().delete()
        self.api_client.force_authenticate(user=self.user)
        response = self.api_client.get(reverse("dashboard_api_jobs"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    # 12. Create job page redirects if not authenticated? (It's public, stays)
    # Instead: GET on an API endpoint that requires POST
    def test_api_login_get_not_allowed(self):
        # The API login view is POST-only, a GET should return 405
        response = self.api_client.get(reverse("dashboard_api_login"))
        self.assertEqual(response.status_code, 405)

    # 13. Test that logout with an invalid token in cookie fails gracefully
    def test_api_logout_invalid_token_cookie(self):
        user = User.objects.create_user(username="invlogout", password="test")
        self.client.force_login(user)  # session login, but DRF ignores it
        self.client.cookies["authToken"] = (
            "invalid-token"  # invalid cookie -> unauthenticated
        )
        response = self.client.post(reverse("dashboard_api_logout"))
        self.assertEqual(response.status_code, 401)  # not logged in via token

    # 14. Test job detail with a run that has no name
    def test_job_detail_unnamed_run(self):
        unnamed_run = TrainingRun.objects.create(
            experiment=self.exp, name="", status="running"  # empty name
        )
        self.client.force_login(self.user)
        url = reverse("dashboard_job_detail", kwargs={"job_id": unnamed_run.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)

    # 15. Test index page when many runs exist (pagination not applied, but view fetches last 20)
    def test_index_many_runs(self):
        # Create 25 runs, index shows only last 20
        for i in range(25):
            TrainingRun.objects.create(
                experiment=self.exp, name=f"Run {i}", status="completed"
            )
        response = self.client.get(reverse("dashboard_index"))
        self.assertEqual(response.status_code, 200)
        # The view passes 'runs' to template, we can't check count easily in rendered HTML,
        # but ensure no error.
        self.assertContains(response, "Transformer Pro")

    # 16. Statistics with multiple days (ensure date grouping works)
    def test_statistics_multi_day(self):
        from datetime import timedelta

        from django.utils import timezone

        today = timezone.now()
        yesterday = today - timedelta(days=1)
        TrainingRun.objects.create(
            experiment=self.exp,
            name="Yesterday Run",
            status="completed",
            created_at=yesterday,
        )
        self.client.force_login(self.user)
        response = self.client.get(reverse("dashboard_statistics"))
        self.assertEqual(response.status_code, 200)
        # The view passes 'days_labels' (as JSON) to template, but our template doesn't show it.
        # We just verify the page loads and contains some expected numbers.
        self.assertContains(response, "success_rate")  # the string is present

    # 17. API register response includes token and cookie (already tested, but verify cookie values)
    def test_api_register_cookie_details(self):
        data = {"username": "cookieuser", "password": "cookiepass"}
        response = self.api_client.post(
            reverse("dashboard_api_register"),
            data=json.dumps(data),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 201)
        # Cookie 'authToken' should exist and have a value
        cookie = response.cookies.get("authToken")
        self.assertIsNotNone(cookie)
        self.assertTrue(cookie.value)

    # 18. API login sets cookie with correct token
    def test_api_login_cookie_token_match(self):
        User.objects.create_user(username="cookiematch", password="test")
        data = {"username": "cookiematch", "password": "test"}
        response = self.api_client.post(
            reverse("dashboard_api_login"),
            data=json.dumps(data),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        token_from_body = json_data["token"]
        cookie_token = response.cookies["authToken"].value
        self.assertEqual(token_from_body, cookie_token)

    # 19. Dashboard API jobs contains correct fields
    def test_dashboard_api_jobs_field_set(self):
        self.api_client.force_authenticate(user=self.user)
        response = self.api_client.get(reverse("dashboard_api_jobs"))
        data = response.json()
        run = data[0]
        expected_fields = {
            "id",
            "status",
            "config",
            "created_at",
            "started_at",
            "completed_at",
            "current_step",
            "total_steps",
            "eta_seconds",
            "gpu_percent",
            "gpu_memory_mb",
            "cpu_percent",
            "ram_mb",
        }
        self.assertTrue(expected_fields.issubset(run.keys()))

    # 20. Test that login page is accessible even after middleware override (already tested)
    # We'll test that the redirect to login sets the next parameter correctly
    def test_redirect_next_parameter_preserved(self):
        from urllib.parse import quote

        target = "/dashboard/statistics/"
        url = f"/dashboard/job/{self.run.id}/?next={quote(target)}"
        response = self.client.get(url)
        # Should redirect to login with next parameter
        self.assertRedirects(response, f"/login/?next={url}", fetch_redirect_response=False)
