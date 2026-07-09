"""REST API client — used as fallback when WebSocket is unavailable."""
import requests, logging

logger = logging.getLogger("neuralops.api")

class APIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Token {api_key}',
            'Content-Type': 'application/json',
        })

    def _post(self, path: str, data: dict) -> dict:
        try:
            resp = self.session.post(f'{self.base_url}{path}', json=data, timeout=30)
            if resp.status_code in (200, 201):
                return resp.json()
            logger.error("API error %d on %s: %s", resp.status_code, path, resp.text)
            return {}
        except requests.RequestException as e:
            logger.error("API call failed %s: %s", path, e)
            return {}

    def _get(self, path: str) -> dict:
        try:
            resp = self.session.get(f'{self.base_url}{path}', timeout=30)
            if resp.status_code == 200:
                return resp.json()
            return {}
        except requests.RequestException as e:
            logger.error("API GET failed %s: %s", path, e)
            return {}

    def create_or_update_listing(self, data: dict) -> dict:
        return self._post('/api/gpu/listings/register/', data)

    def update_listing_status(self, status: str) -> dict:
        return self._post('/api/gpu/listings/update-status/', {'status': status})

    def submit_benchmark(self, listing_id: str, results: dict) -> dict:
        return self._post('/api/gpu/benchmarks/submit/', {
            'listing_id': listing_id, 'results': results
        })

    def report_job_status(self, job_id: str, status: str, description: str = '') -> dict:
        return self._post(f'/api/gpu/jobs/{job_id}/status/', {
            'status': status, 'description': description
        })

    def report_job_completed(self, job_id: str, output_hash: str = '', logs: str = '') -> dict:
        return self._post(f'/api/gpu/jobs/{job_id}/complete/', {
            'output_hash': output_hash, 'logs': logs[-100000:]
        })

    def report_job_failed(self, job_id: str, error: str, logs: str = '') -> dict:
        return self._post(f'/api/gpu/jobs/{job_id}/fail/', {
            'error': error, 'logs': logs[-100000:]
        })

    def report_job_logs(self, job_id: str, logs: str) -> dict:
        return self._post(f'/api/gpu/jobs/{job_id}/logs/', {'logs': logs})