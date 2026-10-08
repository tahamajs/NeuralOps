"""Fetch PyPI download statistics for neuralops."""
import json
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta

def fetch_stats(days: int = 30):
    url = "https://pypistats.org/api/packages/neuralops/recent"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "NeuralOps-Analytics/1.2.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.load(r)
        return data
    except Exception as e:
        print(f"Error fetching stats: {e}")
        return None

if __name__ == "__main__":
    stats = fetch_stats(30)
    if stats:
        print(json.dumps(stats, indent=2))
