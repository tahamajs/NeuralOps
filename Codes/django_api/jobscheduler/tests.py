from django.test import TestCase

# Create your tests here.
def test_scheduler():
    sched = GPUScheduler(total_gpus=2)
    sched.start()
    # Submit jobs with different priorities
    sched.add_job('job1', priority=5, estimated_duration=30)
    sched.add_job('job2', priority=2, estimated_duration=1200)
    sched.add_job('job3', priority=8, estimated_duration=45)
    # Let the scheduler run for a while
    time.sleep(10)
    # Check logs


import requests
import time

API_BASE = "http://localhost:8000/api"
TOKEN = "your-auth-token"

headers = {"Authorization": f"Token {TOKEN}"}

# Submit a few jobs
for i in range(5):
    payload = {
        "name": f"test_job_{i}",
        "model_name": "distilbert-base-uncased",
        "task_name": "mrpc",
        "priority": i+1,
        "estimated_duration": 30 if i < 2 else 300,
        "gpu_requirements": 1,
        "memory_requirements": 4
    }
    resp = requests.post(f"{API_BASE}/jobs/", json=payload, headers=headers)
    print(f"Submitted job {i}: {resp.status_code}")

time.sleep(10)
# Check status
resp = requests.get(f"{API_BASE}/jobs/", headers=headers)
print(resp.json())