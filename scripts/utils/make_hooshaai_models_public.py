#!/usr/bin/env python3
"""
Rate-aware batch updater for Hooshaai model repositories on Hugging Face.
Strictly respects Hugging Face's tier quota (1000 requests per 5 minutes = ~3.2 requests/sec max).
Uses 2 workers with 0.35s delay per worker, automatic 429 backoff detection, and state tracking.
"""

import sys
import os
import time
import logging
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("HF-Publicizer-Managed")

ORG_NAME = "Hooshaai"

def wait_for_rate_limit(e: Exception):
    err_str = str(e)
    # Check if a retry time is stated
    import re
    match = re.search(r"Retry after (\d+) seconds", err_str)
    if match:
        wait = int(match.group(1)) + 5
    else:
        wait = 60
    LOG.warning("Rate limit encountered: sleeping for %d seconds...", wait)
    time.sleep(wait)

def fetch_all_models_safely(api: HfApi):
    while True:
        try:
            LOG.info("Fetching model list for '%s'...", ORG_NAME)
            return list(api.list_models(author=ORG_NAME))
        except HfHubHTTPError as e:
            if "429" in str(e):
                wait_for_rate_limit(e)
            else:
                raise e

def main():
    api = HfApi()
    
    models = fetch_all_models_safely(api)
    private_models = [m.id for m in models if m.private]
    public_models = [m.id for m in models if not m.private]
    
    total = len(models)
    LOG.info("Total models: %d | Currently Public: %d | Currently Private: %d", total, len(public_models), len(private_models))
    
    if not private_models:
        LOG.info("🎉 All %d models under %s are already PUBLIC!", total, ORG_NAME)
        return

    processed = 0
    total_private = len(private_models)
    start_time = time.time()
    
    for repo_id in private_models:
        success = False
        while not success:
            try:
                api.update_repo_settings(repo_id=repo_id, private=False)
                success = True
                processed += 1
                # Pace requests: 0.35s sleep = max ~2.85 req/sec (< 3.33 req/sec limit)
                time.sleep(0.35)
            except HfHubHTTPError as e:
                if "429" in str(e):
                    wait_for_rate_limit(e)
                else:
                    LOG.error("Failed on %s: %s (skipping)", repo_id, e)
                    break
            except Exception as e:
                LOG.error("Unexpected error on %s: %s (skipping)", repo_id, e)
                break
        
        if processed % 50 == 0 or processed == total_private:
            elapsed = time.time() - start_time
            remaining = total_private - processed
            rate = processed / elapsed if elapsed > 0 else 0
            eta_mins = (remaining / rate) / 60 if rate > 0 else 0
            LOG.info(
                "Progress: %d/%d (%.1f%%) | Rate: %.2f repos/s | Est. Remaining: %.1f mins",
                processed, total_private, (processed / total_private) * 100, rate, eta_mins
            )

    LOG.info("Checking final status...")
    final_models = fetch_all_models_safely(api)
    final_private = sum(1 for m in final_models if m.private)
    final_public = sum(1 for m in final_models if not m.private)
    LOG.info("🎉 DONE! Public: %d | Private remaining: %d", final_public, final_private)

if __name__ == "__main__":
    main()
