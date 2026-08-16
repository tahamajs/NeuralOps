"""
End-to-End Integration & Launch Flow Debugger Test Suite.
Tests full user journeys across 4 major flows:
  - Flow 1: User Registration & Authentication -> Token storage -> Profile load
  - Flow 2: Job Creation -> Configuration of attention architecture (NASC / AEGIS / SVD) -> Launch job via API
  - Flow 3: Live Run Monitoring -> Real-time metric streaming -> Logs via WebSocket terminal
  - Flow 4: Export Artifacts -> Generate Model Card / LaTeX report
"""

import json
import uuid
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from channels.testing import WebsocketCommunicator
from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import TransactionTestCase
from django_api.asgi import application
from experiment.models import Experiment, TrainingRun, TrainingSnapshot
from profiles.models import Profile
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient


class EndToEndIntegrationFlowTests(TransactionTestCase):
    """Full End-to-End Integration Test Suite for NeuralOps."""

    def setUp(self):
        cache.clear()
        self.client = APIClient()

    # ═══════════════════════════════════════════════════════════════════
    # Flow 1: User Registration & Auth -> Token Storage -> Profile Load
    # ═══════════════════════════════════════════════════════════════════
    def test_flow1_user_registration_auth_and_profile_lifecycle(self):
        """Test user registration, login, token retrieval, user info, and profile loading."""
        # 1. User Registration
        reg_payload = {
            "username": "quantum_researcher",
            "email": "quantum@neuralops.io",
            "password": "SecurePassword123!",
        }
        reg_resp = self.client.post("/api/auth/register/", reg_payload, format="json")
        self.assertEqual(reg_resp.status_code, 201)
        self.assertIn("token", reg_resp.data)
        self.assertEqual(reg_resp.data["username"], "quantum_researcher")
        token = reg_resp.data["token"]
        self.assertTrue(reg_resp.cookies.get("authToken"))

        # 2. User Login
        login_payload = {
            "username": "quantum_researcher",
            "password": "SecurePassword123!",
        }
        login_resp = self.client.post("/api/auth/login/", login_payload, format="json")
        self.assertEqual(login_resp.status_code, 200)
        self.assertEqual(login_resp.data["token"], token)

        # 3. Authenticated client setup
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {token}")

        # 4. User Info load
        user_resp = self.client.get("/api/auth/user/")
        self.assertEqual(user_resp.status_code, 200)
        self.assertEqual(user_resp.data["username"], "quantum_researcher")
        self.assertEqual(user_resp.data["email"], "quantum@neuralops.io")

        # 5. User Profile load (/profiles/me/)
        profile_resp = self.client.get("/api/profiles/me/")
        self.assertEqual(profile_resp.status_code, 200)
        self.assertIn("user", profile_resp.data)

        # 6. User Profile update
        update_payload = {
            "bio": "Deep Learning Scientist specializing in efficient attention architectures.",
            "affiliation": "NeuralOps Research Lab",
        }
        patch_resp = self.client.patch("/api/profiles/me/", update_payload, format="json")
        self.assertEqual(patch_resp.status_code, 200)
        self.assertEqual(patch_resp.data["bio"], update_payload["bio"])
        self.assertEqual(patch_resp.data["affiliation"], update_payload["affiliation"])

        # 7. User Profile lookup by username identifier
        user_lookup = self.client.get("/api/profiles/user/quantum_researcher/")
        self.assertEqual(user_lookup.status_code, 200)
        self.assertEqual(user_lookup.data["bio"], update_payload["bio"])

    # ═══════════════════════════════════════════════════════════════════
    # Flow 2: Job Creation -> Configuration of Attention -> Launch API
    # ═══════════════════════════════════════════════════════════════════
    def test_flow2_job_creation_attention_config_and_launch(self):
        """Test configuration discovery, attention selection (NASC/AEGIS/SVD), and job launch."""
        user = User.objects.create_user(username="arch_dev", password="password123")
        token = Token.objects.create(user=user)
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {token.key}")

        # 1. Fetch Attention architectures list
        attn_resp = self.client.get("/api/config/attentions/")
        self.assertEqual(attn_resp.status_code, 200)
        attentions = attn_resp.data.get("attentions", [])
        self.assertTrue(len(attentions) > 0)
        # Check presence of key efficient attention families
        has_nasc = any("nasc" in a.lower() for a in attentions)
        has_aegis = any("aegis" in a.lower() for a in attentions)
        has_svd = any("svd" in a.lower() for a in attentions)
        self.assertTrue(has_nasc or has_aegis or has_svd)

        # 2. Fetch Models and Methods
        models_resp = self.client.get("/api/config/models/")
        self.assertEqual(models_resp.status_code, 200)
        self.assertIn("models", models_resp.data)

        methods_resp = self.client.get("/api/config/methods/")
        self.assertEqual(methods_resp.status_code, 200)
        self.assertIn("methods", methods_resp.data)

        # 3. Launch Training Job with NASC Attention Architecture
        nasc_job_config = {
            "name": "NASC-GLUE-MRPC-Benchmark",
            "model_name": "bert-base-uncased",
            "task_name": "mrpc",
            "attention_type": "nasc",
            "method": "svd",
            "compression_rank": 64,
            "epochs": 3,
            "batch_size": 16,
            "learning_rate": 3e-5,
            "max_seq_length": 128,
            "fp16": True,
            "seed": 42,
        }
        train_launch = self.client.post("/api/experiments/start_train/", nasc_job_config, format="json")
        self.assertEqual(train_launch.status_code, 201)
        self.assertIn("id", train_launch.data)
        run_id = train_launch.data["id"]
        self.assertEqual(train_launch.data["status"], "queued")
        self.assertEqual(train_launch.data["config"]["attention_type"], "nasc")

        # 4. Launch Evaluation Job with AEGIS Attention Architecture
        aegis_job_config = {
            "name": "AEGIS-V6-SST2-Eval",
            "model_name": "distilbert-base-uncased",
            "task_name": "sst2",
            "attention_type": "aegis_v6",
            "method": "baseline",
            "compression_rank": 32,
            "epochs": 1,
            "batch_size": 32,
        }
        eval_launch = self.client.post("/api/experiments/start_evaluate/", aegis_job_config, format="json")
        self.assertEqual(eval_launch.status_code, 201)
        self.assertIn("id", eval_launch.data)
        self.assertEqual(eval_launch.data["config"]["attention_type"], "aegis_v6")

        # 5. Launch Compression Job with SVD
        svd_job_config = {
            "name": "SVD-Linear-Compression",
            "model_name": "roberta-base",
            "task_name": "mnli",
            "attention_type": "svd",
            "method": "svd",
            "compression_rank": 48,
        }
        comp_launch = self.client.post("/api/experiments/start_compress/", svd_job_config, format="json")
        self.assertEqual(comp_launch.status_code, 201)
        self.assertIn("id", comp_launch.data)

        # 6. Verify run details endpoint
        run_detail = self.client.get(f"/api/runs/{run_id}/")
        self.assertEqual(run_detail.status_code, 200)
        self.assertEqual(run_detail.data["name"], "NASC-GLUE-MRPC-Benchmark")

    # ═══════════════════════════════════════════════════════════════════
    # Flow 3: Live Run Monitoring -> Real-time metric streaming & Terminal WS
    # ═══════════════════════════════════════════════════════════════════
    def test_flow3_live_monitoring_metrics_and_terminal_websocket(self):
        """Test WebSocket real-time metric broadcasting and terminal streaming."""
        user = User.objects.create_user(username="monitor_eng", password="password123")
        token = Token.objects.create(user=user)
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {token.key}")

        exp = Experiment.objects.create(name="LiveMonitorExp")
        run = TrainingRun.objects.create(
            experiment=exp,
            name="LiveRun_01",
            status="running",
            config={"model_name": "distilbert", "attention_type": "nasc"},
        )
        run_id = str(run.id)

        # 1. Test WebSocket connection to JobConsumer (/ws/experiments/<run_id>/)
        async def _test_job_ws():
            communicator = WebsocketCommunicator(
                application, f"/ws/experiments/{run_id}/"
            )
            connected, _ = await communicator.connect()
            self.assertTrue(connected)

            # Broadcast live metric update via channel layer
            channel_layer = get_channel_layer()
            sample_metric_payload = {
                "type": "job_update",
                "data": {
                    "step": 100,
                    "train_loss": 0.3421,
                    "val_accuracy": 0.9120,
                    "throughput_tokens_per_sec": 1420.5,
                    "gpu_memory_used_mb": 4250,
                },
            }
            await channel_layer.group_send(f"job_{run_id}", sample_metric_payload)

            response = await communicator.receive_json_from()
            self.assertEqual(response["step"], 100)
            self.assertEqual(response["train_loss"], 0.3421)
            self.assertEqual(response["val_accuracy"], 0.9120)

            await communicator.disconnect()

        async_to_sync(_test_job_ws)()

        # 2. Test WebSocket connection to TerminalConsumer (/ws/terminal/)
        async def _test_terminal_ws():
            communicator = WebsocketCommunicator(
                application, "/ws/terminal/"
            )
            connected, _ = await communicator.connect()
            self.assertTrue(connected)

            # Check receive or disconnect
            await communicator.disconnect()

        async_to_sync(_test_terminal_ws)()

    # ═══════════════════════════════════════════════════════════════════
    # Flow 4: Export Artifacts -> Generate Model Card / LaTeX Report
    # ═══════════════════════════════════════════════════════════════════
    def test_flow4_export_artifacts_model_card_and_latex_report(self):
        """Test generating structured Model Card and LaTeX report for publication."""
        user = User.objects.create_user(username="author_user", password="password123")
        token = Token.objects.create(user=user)
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {token.key}")

        exp = Experiment.objects.create(name="SpectralAttentionPaper")
        run = TrainingRun.objects.create(
            experiment=exp,
            name="NASC_MRPC_SOTA",
            status="completed",
            config={
                "model_name": "bert-base-uncased",
                "task_name": "mrpc",
                "attention_type": "nasc",
                "compression_rank": 64,
                "learning_rate": 2e-5,
                "epochs": 5,
                "batch_size": 16,
            },
            final_metrics={
                "eval_accuracy": 0.9145,
                "eval_f1": 0.9382,
                "eval_loss": 0.2214,
            },
        )
        run_id = str(run.id)

        # 1. Generate Model Card via API
        card_resp = self.client.get(f"/api/runs/{run_id}/model-card/")
        self.assertEqual(card_resp.status_code, 200)
        self.assertIn("model_card_md", card_resp.data)
        model_card = card_resp.data["model_card_md"]
        self.assertIn("Model Card", model_card)
        self.assertIn("bert-base-uncased", model_card)
        self.assertIn("mrpc", model_card)

        # 2. Generate LaTeX Report via Paper API
        paper_resp = self.client.get(f"/api/experiments/runs/{run_id}/paper/")
        self.assertEqual(paper_resp.status_code, 200)
        self.assertIn("latex", paper_resp.data)
        latex_doc = paper_resp.data["latex"]
        self.assertIn("\\documentclass", latex_doc)
        self.assertIn("bert-base-uncased", latex_doc)
        self.assertIn("nasc", latex_doc)
        self.assertIn("\\begin{document}", latex_doc)
        self.assertIn("\\end{document}", latex_doc)

        # 3. Test LaTeX Download (.tex file)
        download_resp = self.client.get(f"/api/experiments/runs/{run_id}/paper/?download=true")
        self.assertEqual(download_resp.status_code, 200)
        self.assertEqual(download_resp["Content-Type"], "application/x-tex")
        self.assertIn(f"paper_{run_id}.tex", download_resp["Content-Disposition"])

        # 4. Test Experiment Report Export (PDF / HTML view)
        export_resp = self.client.get(f"/runs/{run_id}/export-report-pdf/")
        self.assertEqual(export_resp.status_code, 200)
        self.assertTrue(
            "application/pdf" in export_resp["Content-Type"] or "text/html" in export_resp["Content-Type"]
        )
