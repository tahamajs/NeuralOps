#!/usr/bin/env python
"""
Comprehensive API Endpoint Tester
Tests all backend endpoints and validates responses
"""
import requests
import json
import time
import sys
from datetime import datetime
from typing import Dict, Tuple, Any

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000/api"):
        self.base_url = base_url
        self.token = None
        self.session = requests.Session()
        # Avoid system proxy settings that can cause local 502s.
        self.session.trust_env = False
        self.results = []
        self.test_user = {
            "username": f"testuser_{int(time.time())}",
            "password": "TestPassword123!",
            "email": f"test_{int(time.time())}@test.com"
        }

    def log_result(self, endpoint: str, method: str, status: str, response_code: int = None, message: str = ""):
        """Log test result"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "method": method,
            "status": status,
            "response_code": response_code,
            "message": message
        }
        self.results.append(result)
        status_symbol = "✓" if status == "PASS" else "✗"
        print(f"{status_symbol} [{method:6}] {endpoint:40} [{response_code}] {message}")

    def test_endpoint(self, method: str, endpoint: str, data: Dict = None, 
                     expected_status_code: int = 200, require_auth: bool = True) -> Tuple[bool, Any]:
        """Test a single endpoint"""
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        if require_auth and self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        
        try:
            if method == "GET":
                response = self.session.get(url, headers=headers, timeout=5)
            elif method == "POST":
                response = self.session.post(url, json=data, headers=headers, timeout=5)
            elif method == "PATCH":
                response = self.session.patch(url, json=data, headers=headers, timeout=5)
            elif method == "DELETE":
                response = self.session.delete(url, headers=headers, timeout=5)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            success = response.status_code == expected_status_code
            status = "PASS" if success else "FAIL"
            message = response.status_code != expected_status_code and f"Expected {expected_status_code}" or "OK"

            # If response contains a token, update the stored token for subsequent requests
            try:
                body = response.json()
                if isinstance(body, dict) and body.get('token'):
                    self.token = body.get('token')
            except Exception:
                body = None

            self.log_result(endpoint, method, status, response.status_code, message)

            try:
                return success, response.json()
            except:
                return success, response.text
        except Exception as e:
            self.log_result(endpoint, method, "ERROR", 0, str(e)[:50])
            return False, None

    def run_all_tests(self):
        """Run all endpoint tests"""
        print("\n" + "="*100)
        print("COMPREHENSIVE API ENDPOINT TEST".center(100))
        print("="*100 + "\n")

        # Test 1: Health Check
        print("\n[1] HEALTH & VERSION ENDPOINTS")
        print("-" * 100)
        self.test_endpoint("GET", "/health/", require_auth=False)
        self.test_endpoint("GET", "/version/", require_auth=False)
        self.test_endpoint("GET", "/system/status/", require_auth=False)
        self.test_endpoint("GET", "/system/gpu/", require_auth=False)

        # Test 2: Authentication
        print("\n[2] AUTHENTICATION ENDPOINTS")
        print("-" * 100)
        success, response = self.test_endpoint("POST", "/auth/register/", self.test_user, 
                                              expected_status_code=201, require_auth=False)
        if success and response and isinstance(response, dict):
            self.token = response.get('token')
            print(f"   └─ Token obtained: {self.token[:20]}...")

        self.test_endpoint("POST", "/auth/login/", self.test_user, 
                          expected_status_code=200, require_auth=False)
        
        self.test_endpoint("GET", "/auth/user/", expected_status_code=200)
        self.test_endpoint("PATCH", "/auth/profile/", 
                          {"first_name": "Test", "last_name": "User"}, expected_status_code=200)
        self.test_endpoint("POST", "/auth/change-password/", 
                          {"old_password": self.test_user["password"], "new_password": "NewPass123!"}, 
                          expected_status_code=200)
        self.test_endpoint("POST", "/auth/refresh-token/", expected_status_code=200)

        # Test 3: Configuration Endpoints
        print("\n[3] CONFIGURATION ENDPOINTS")
        print("-" * 100)
        self.test_endpoint("GET", "/config/attentions/", expected_status_code=200)
        self.test_endpoint("GET", "/config/compressors/", expected_status_code=200)
        self.test_endpoint("GET", "/config/models/", expected_status_code=200)
        self.test_endpoint("GET", "/config/tasks/", expected_status_code=200)
        self.test_endpoint("GET", "/config/methods/", expected_status_code=200)
        self.test_endpoint("GET", "/config/presets/", expected_status_code=200)

        # Test 4: Experiment Endpoints
        print("\n[4] EXPERIMENT ENDPOINTS")
        print("-" * 100)
        exp_id = None
        success, response = self.test_endpoint("GET", "/experiments/", expected_status_code=200)
        
        success, response = self.test_endpoint("POST", "/experiments/", 
                          {"name": "Test Experiment", "description": "API Test"}, 
                          expected_status_code=201)
        if success and response and isinstance(response, dict):
            exp_id = response.get('id')
            print(f"   └─ Experiment created: {exp_id}")

        if exp_id:
            self.test_endpoint("GET", f"/experiments/{exp_id}/", expected_status_code=200)
            self.test_endpoint("PATCH", f"/experiments/{exp_id}/", 
                              {"name": "Updated Test Experiment"}, expected_status_code=200)

        # Test 5: Training/Evaluation/Compression Endpoints
        print("\n[5] JOB ENDPOINTS (Training/Evaluation/Compression)")
        print("-" * 100)
        
        train_config = {
            "name": "Test Training",
            "model": "distilbert-base-uncased",
            "task": "sst2",
            "num_epochs": 1,
            "batch_size": 8
        }
        success, response = self.test_endpoint("POST", "/train/", train_config, 
                                              expected_status_code=201)
        run_id = response.get('id') if success and isinstance(response, dict) else None
        if run_id:
            print(f"   └─ Training run created: {run_id}")
            self.test_endpoint("GET", f"/runs/{run_id}/", expected_status_code=200)

        eval_config = {
            "name": "Test Evaluation",
            "model": "distilbert-base-uncased",
            "task": "sst2"
        }
        self.test_endpoint("POST", "/evaluate/", eval_config, expected_status_code=201)

        compress_config = {
            "name": "Test Compression",
            "model": "distilbert-base-uncased",
            "task": "sst2",
            "compression_method": "svd"
        }
        self.test_endpoint("POST", "/compress/", compress_config, expected_status_code=201)

        if run_id:
            self.test_endpoint("POST", f"/runs/{run_id}/stop/", expected_status_code=200)

        # Test 6: Cleanup
        print("\n[6] CLEANUP")
        print("-" * 100)
        if exp_id:
            self.test_endpoint("DELETE", f"/experiments/{exp_id}/", expected_status_code=204)
        
        self.test_endpoint("POST", "/auth/logout/", expected_status_code=200)

        # Print Summary
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*100)
        print("TEST SUMMARY".center(100))
        print("="*100)
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r['status'] == 'PASS')
        failed = sum(1 for r in self.results if r['status'] == 'FAIL')
        errors = sum(1 for r in self.results if r['status'] == 'ERROR')
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed} ✓")
        print(f"Failed: {failed} ✗")
        print(f"Errors: {errors} ⚠")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        
        if failed > 0 or errors > 0:
            print("\nFailed/Error Tests:")
            for result in self.results:
                if result['status'] in ['FAIL', 'ERROR']:
                    print(f"  - [{result['method']}] {result['endpoint']}: {result['message']}")
        
        print("\n" + "="*100 + "\n")
        
        return passed, failed, errors

    def export_results(self, filename: str = "test_results.json"):
        """Export test results to JSON"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results exported to {filename}")

if __name__ == "__main__":
    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/api"
    
    print(f"\nTesting API at: {base_url}\n")
    
    tester = APITester(base_url)
    try:
        tester.run_all_tests()
        tester.export_results()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        sys.exit(1)
