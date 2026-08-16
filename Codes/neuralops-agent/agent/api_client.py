"""
REST API client for NeuralOps GPU Provider Agent.
Handles communication with Django backend for registration, heartbeats,
benchmarks, job status transitions, and authentication token rotation.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union

import requests

logger = logging.getLogger("neuralops.api")


class APIClient:
    """REST client for NeuralOps API endpoints with retries and token rotation."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = 30,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        on_token_refreshed: Optional[Callable[[str], None]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.on_token_refreshed = on_token_refreshed

        self.session = requests.Session()
        self._update_auth_header()

    def _update_auth_header(self):
        """Update session headers with current authorization token."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Token {self.api_key}"
        self.session.headers.update(headers)

    def set_api_key(self, api_key: str):
        """Update API key dynamically (e.g. after token rotation)."""
        self.api_key = api_key
        self._update_auth_header()
        if self.on_token_refreshed:
            try:
                self.on_token_refreshed(api_key)
            except Exception as e:
                logger.warning("Token refresh callback error: %s", e)

    def _request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        fallback_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute an HTTP request with retry logic and path fallback support.
        """
        paths_to_try = [path]
        if fallback_paths:
            paths_to_try.extend(fallback_paths)

        last_error = None
        for current_path in paths_to_try:
            url = f"{self.base_url}{current_path}"
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = self.session.request(
                        method=method,
                        url=url,
                        json=data if method in ("POST", "PUT", "PATCH") else None,
                        params=params,
                        timeout=self.timeout,
                    )

                    if resp.status_code in (200, 201, 204):
                        if resp.status_code == 204 or not resp.content:
                            return {"status": "ok", "status_code": resp.status_code}
                        try:
                            return resp.json()
                        except ValueError:
                            return {"status": "ok", "text": resp.text}

                    # Handle 404 by trying next candidate path if available
                    if resp.status_code == 404 and len(paths_to_try) > 1 and current_path != paths_to_try[-1]:
                        logger.debug("Path %s returned 404, attempting fallback...", current_path)
                        break

                    # Handle 401 Unauthorized (can happen when token expired)
                    if resp.status_code == 401:
                        logger.error("API 401 Unauthorized on %s: %s", current_path, resp.text)
                        return {"error": "unauthorized", "status_code": 401, "detail": resp.text}

                    # Handle 5xx server errors with retry backoff
                    if resp.status_code >= 500 and attempt < self.max_retries:
                        sleep_time = self.retry_backoff * (2 ** (attempt - 1))
                        logger.warning(
                            "API error %d on %s (attempt %d/%d). Retrying in %.2fs...",
                            resp.status_code,
                            current_path,
                            attempt,
                            self.max_retries,
                            sleep_time,
                        )
                        time.sleep(sleep_time)
                        continue

                    logger.error(
                        "API HTTP error %d on %s: %s",
                        resp.status_code,
                        current_path,
                        resp.text,
                    )
                    return {"error": f"HTTP {resp.status_code}", "status_code": resp.status_code, "detail": resp.text}

                except (requests.ConnectionError, requests.Timeout) as e:
                    last_error = e
                    if attempt < self.max_retries:
                        sleep_time = self.retry_backoff * (2 ** (attempt - 1))
                        logger.warning(
                            "Network %s on %s (attempt %d/%d). Retrying in %.2fs...",
                            type(e).__name__,
                            current_path,
                            attempt,
                            self.max_retries,
                            sleep_time,
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.error("API call failed after %d retries on %s: %s", self.max_retries, current_path, e)
                except requests.RequestException as e:
                    logger.error("API request exception on %s: %s", current_path, e)
                    return {"error": str(e)}

        return {"error": str(last_error) if last_error else "request_failed"}

    def _post(self, path: str, data: Dict[str, Any], fallback_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        return self._request("POST", path, data=data, fallback_paths=fallback_paths)

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None, fallback_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        return self._request("GET", path, params=params, fallback_paths=fallback_paths)

    # ── Authentication & Token Rotation ─────────────────────────

    def rotate_token(self) -> Optional[str]:
        """
        Request a new API token from backend and update local session headers.
        Aligned with Django `/api/auth/refresh-token/`.
        """
        resp = self._post(
            "/api/auth/refresh-token/",
            data={},
            fallback_paths=["/auth/refresh-token/", "/api/auth/token/refresh/"],
        )
        new_token = resp.get("token")
        if new_token:
            logger.info("Successfully rotated API token")
            self.set_api_key(new_token)
            return new_token
        logger.error("Failed to rotate token: %s", resp)
        return None

    # ── GPU Listings & Registration ─────────────────────────────

    def create_or_update_listing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register or update a GPU listing on the marketplace.
        Primary: /api/gpu/listings/register/
        Fallback: /api/listings/register/, /api/marketplace/listings/register/
        """
        return self._post(
            "/api/gpu/listings/register/",
            data,
            fallback_paths=[
                "/api/listings/register/",
                "/api/marketplace/listings/register/",
            ],
        )

    def update_listing_status(self, status: str, listing_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Update the online/offline/busy status of the provider listing.
        Primary: /api/gpu/listings/update-status/
        Fallback: /api/listings/update-status/
        """
        payload = {"status": status}
        if listing_id:
            payload["listing_id"] = listing_id
        return self._post(
            "/api/gpu/listings/update-status/",
            payload,
            fallback_paths=["/api/listings/update-status/"],
        )

    def send_heartbeat(self, listing_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Send REST heartbeat ping to refresh provider listing active timestamp.
        Primary: /api/gpu/listings/heartbeat/
        Fallback: /api/listings/heartbeat/
        """
        payload = {"listing_id": listing_id} if listing_id else {}
        return self._post(
            "/api/gpu/listings/heartbeat/",
            payload,
            fallback_paths=["/api/listings/heartbeat/"],
        )

    def get_listing(self, listing_id: str) -> Dict[str, Any]:
        """Retrieve details of a specific GPU listing."""
        return self._get(
            f"/api/gpu/listings/{listing_id}/",
            fallback_paths=[f"/api/listings/{listing_id}/"],
        )

    def get_my_listings(self) -> Union[Dict[str, Any], List[Any]]:
        """Retrieve all listings owned by authenticated provider."""
        return self._get(
            "/api/gpu/listings/mine/",
            fallback_paths=["/api/listings/mine/"],
        )

    # ── Benchmarking ────────────────────────────────────────────

    def submit_benchmark(self, listing_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit benchmark results for a GPU listing.
        Primary: /api/gpu/benchmarks/submit/
        Fallback: /api/benchmarks/submit/
        """
        return self._post(
            "/api/gpu/benchmarks/submit/",
            {"listing_id": listing_id, "results": results},
            fallback_paths=["/api/benchmarks/submit/"],
        )

    # ── Jobs Lifecycle ──────────────────────────────────────────

    def get_jobs(self, status: Optional[str] = None) -> Dict[str, Any]:
        """Fetch list of compute jobs assigned to this provider."""
        params = {"status": status} if status else None
        return self._get(
            "/api/gpu/jobs/",
            params=params,
            fallback_paths=["/api/jobs/"],
        )

    def get_job_detail(self, job_id: str) -> Dict[str, Any]:
        """Get details for a specific compute job."""
        return self._get(
            f"/api/gpu/jobs/{job_id}/",
            fallback_paths=[f"/api/jobs/{job_id}/"],
        )

    def report_job_status(
        self,
        job_id: str,
        status: str,
        description: str = "",
        container_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Report job status transition (e.g., provisioning, running).
        Primary: /api/gpu/jobs/{job_id}/status/
        Fallback: /api/jobs/{job_id}/status/
        """
        payload = {"status": status, "description": description}
        if container_id:
            payload["container_id"] = container_id
        return self._post(
            f"/api/gpu/jobs/{job_id}/status/",
            payload,
            fallback_paths=[f"/api/jobs/{job_id}/status/"],
        )

    def report_job_completed(
        self,
        job_id: str,
        output_hash: str = "",
        output_data_url: str = "",
        logs: str = "",
    ) -> Dict[str, Any]:
        """
        Report that a compute job finished successfully.
        Primary: /api/gpu/jobs/{job_id}/complete/
        Fallback: /api/jobs/{job_id}/complete/
        """
        payload = {
            "output_hash": output_hash,
            "output_data_url": output_data_url or output_hash,
            "logs": logs[-100000:] if logs else "",
        }
        return self._post(
            f"/api/gpu/jobs/{job_id}/complete/",
            payload,
            fallback_paths=[f"/api/jobs/{job_id}/complete/"],
        )

    def report_job_failed(
        self,
        job_id: str,
        error: str,
        logs: str = "",
    ) -> Dict[str, Any]:
        """
        Report that a compute job failed with an error message.
        Primary: /api/gpu/jobs/{job_id}/fail/
        Fallback: /api/jobs/{job_id}/fail/
        """
        payload = {
            "error": error,
            "error_message": error,
            "logs": logs[-100000:] if logs else "",
        }
        return self._post(
            f"/api/gpu/jobs/{job_id}/fail/",
            payload,
            fallback_paths=[f"/api/jobs/{job_id}/fail/"],
        )

    def report_job_logs(self, job_id: str, logs: str) -> Dict[str, Any]:
        """
        Stream or append logs for a running compute job.
        Primary: /api/gpu/jobs/{job_id}/logs/
        Fallback: /api/jobs/{job_id}/logs/
        """
        return self._post(
            f"/api/gpu/jobs/{job_id}/logs/",
            {"logs": logs},
            fallback_paths=[f"/api/jobs/{job_id}/logs/"],
        )

    # ── Auto-Update Support ─────────────────────────────────────

    def check_for_updates(self, current_version: str = "0.0.0") -> Dict[str, Any]:
        """Check whether a newer version of the agent is available."""
        return self._post(
            "/api/gpu/agent/check-update/",
            {"version": current_version},
            fallback_paths=["/api/agent/check-update/"],
        )

    def get_latest_version(
        self, platform: str = "linux", current_version: str = "0.0.0"
    ) -> Dict[str, Any]:
        """Query latest release details for binary download."""
        return self._get(
            "/api/gpu/agent/latest/",
            params={"platform": platform, "current": current_version},
            fallback_paths=["/api/agent/latest/"],
        )
