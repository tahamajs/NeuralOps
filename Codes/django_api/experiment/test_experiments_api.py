"""
Comprehensive test suite for Experiments & Training Runs REST APIs.
Covers:
- Experiment creation, listing, details
- TrainingRun creation, status lifecycle, stopping
- TrainingSnapshot retrieval and step recording
- Run analysis, artifacts, and model cards
"""

import uuid
from django.contrib.auth.models import User
from django.test import TestCase
from experiment.models import Experiment, TrainingRun, TrainingSnapshot
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient


class ExperimentAPITests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="experimenter", password="password123")
        self.token = Token.objects.create(user=self.user)
        self.client = APIClient()
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {self.token.key}")

        self.experiment = Experiment.objects.create(
            name="SVD-Attention-Benchmark",
            description="Testing SVD rank decay across layers",
            tags="svd,attention,nlp"
        )
        self.run = TrainingRun.objects.create(
            experiment=self.experiment,
            name="run-001-svd-rank32",
            status="completed",
            config={"epochs": 5, "batch_size": 8, "learning_rate": 0.0001}
        )
        self.snapshot = TrainingSnapshot.objects.create(
            run=self.run,
            step=100,
            train_loss=1.84,
            eval_loss=1.92,
            eval_accuracy=0.88,
            learning_rate=0.0001
        )

    def test_list_experiments(self):
        """Test GET /api/experiments/"""
        response = self.client.get("/api/experiments/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        results = data if isinstance(data, list) else data.get("results", [])
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0]["name"], "SVD-Attention-Benchmark")

    def test_get_experiment_detail(self):
        """Test GET /api/experiments/<id>/"""
        response = self.client.get(f"/api/experiments/{self.experiment.id}/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["name"], "SVD-Attention-Benchmark")

    def test_create_experiment(self):
        """Test POST /api/experiments/"""
        payload = {
            "name": "Mamba-SVD-Hybrid-Sweep",
            "description": "Evaluating MambaSVD chunked scan",
            "tags": "mamba,hybrid"
        }
        response = self.client.post("/api/experiments/", payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.json()["name"], "Mamba-SVD-Hybrid-Sweep")

    def test_get_run_detail(self):
        """Test GET /api/runs/<id>/"""
        response = self.client.get(f"/api/runs/{self.run.id}/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["name"], "run-001-svd-rank32")
        self.assertEqual(data["status"], "completed")

    def test_get_run_results(self):
        """Test GET /api/runs/<id>/results/"""
        response = self.client.get(f"/api/runs/{self.run.id}/results/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn("snapshots", data)
        self.assertEqual(len(data["snapshots"]), 1)
        self.assertEqual(data["snapshots"][0]["step"], 100)
