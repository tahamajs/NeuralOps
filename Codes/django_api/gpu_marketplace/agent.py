"""
Provider-side agent — polls for jobs and executes them in Docker containers.

This is a lightweight Python script that providers run on their machines.
It registers with the platform, polls for new jobs, and executes them.

Usage:
    python -m gpu_marketplace.agent --api-key <key> --api-url <url>
"""

import subprocess
import time

import requests


class GPUProviderAgent:
    """Agent that runs on the provider's machine."""

    def __init__(self, api_key: str, api_url: str, listing_id: str):
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.listing_id = listing_id
        self.headers = {"Authorization": f"Token {api_key}"}

    def heartbeat(self):
        """Send heartbeat to platform."""
        try:
            requests.post(
                f"{self.api_url}/api/gpu/listings/{self.listing_id}/heartbeat/",
                headers=self.headers,
                json={"status": "online"},
                timeout=5,
            )
        except Exception:
            pass

    def poll_jobs(self):
        """Poll for pending jobs assigned to this listing."""
        resp = requests.get(
            f"{self.api_url}/api/gpu/jobs/",
            headers=self.headers,
            params={"status": "queued", "listing": self.listing_id},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            for job in data.get("results", []):
                self.execute_job(job)

    def execute_job(self, job: dict):
        """Execute a compute job using Docker."""
        job_id = job["id"]
        image = job.get("docker_image", "pytorch/pytorch:latest")
        command = job.get("command", "")

        # Pull image
        subprocess.run(["docker", "pull", image], check=False)

        # Run container
        container_id = (
            subprocess.check_output(
                [
                    "docker",
                    "run",
                    "-d",
                    "--gpus",
                    "all",
                    "--name",
                    f"neuralops-job-{job_id}",
                    image,
                    "bash",
                    "-c",
                    command,
                ]
            )
            .decode()
            .strip()
        )

        # Mark running
        requests.post(
            f"{self.api_url}/api/gpu/jobs/{job_id}/start/",
            headers=self.headers,
            json={"container_id": container_id},
            timeout=5,
        )

    def run(self, poll_interval: int = 30):
        """Main loop: heartbeat + poll + execute."""
        print(f"Agent started for listing {self.listing_id}")
        while True:
            try:
                self.heartbeat()
                self.poll_jobs()
            except Exception as e:
                print(f"Error: {e}")
            time.sleep(poll_interval)
