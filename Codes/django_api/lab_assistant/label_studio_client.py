"""Label Studio API Client — integrates with an external Label Studio instance."""

import logging

import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class LabelStudioClient:
    """Client for Label Studio REST API."""

    def __init__(self, base_url=None, api_key=None):
        self.base_url = base_url or getattr(
            settings, "LABEL_STUDIO_URL", "http://localhost:8080"
        )
        self.api_key = api_key or getattr(settings, "LABEL_STUDIO_API_KEY", "")
        self.headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

    def create_project(self, name, description, label_config):
        """Create a new Label Studio project."""
        resp = requests.post(
            f"{self.base_url}/api/projects",
            headers=self.headers,
            json={
                "title": name,
                "description": description,
                "label_config": label_config,
            },
        )
        if resp.status_code == 201:
            return resp.json()
        logger.error(f"Failed to create Label Studio project: {resp.text}")
        return None

    def import_tasks(self, project_id, tasks):
        """Import tasks into a Label Studio project."""
        resp = requests.post(
            f"{self.base_url}/api/projects/{project_id}/import",
            headers=self.headers,
            json=tasks,
        )
        return resp.status_code == 201

    def get_project_stats(self, project_id):
        """Get annotation statistics for a project."""
        resp = requests.get(
            f"{self.base_url}/api/projects/{project_id}",
            headers=self.headers,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "total_tasks": data.get("task_number", 0),
                "completed_tasks": data.get("total_annotations_number", 0),
            }
        return None

    def get_annotations(self, project_id, task_id):
        """Get annotations for a specific task."""
        resp = requests.get(
            f"{self.base_url}/api/tasks/{task_id}/annotations",
            headers=self.headers,
        )
        if resp.status_code == 200:
            return resp.json()
        return []

    def create_prediction(self, task_id, result, score=None, model_version=None):
        """Submit an AI prediction for a task."""
        payload = {
            "result": result,
            "score": score,
            "model_version": model_version or "neuralops-ai-v1",
        }
        resp = requests.post(
            f"{self.base_url}/api/tasks/{task_id}/predictions",
            headers=self.headers,
            json=payload,
        )
        return resp.status_code == 201
